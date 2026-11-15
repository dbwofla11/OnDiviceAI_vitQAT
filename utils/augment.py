"""학습용 데이터 증강을 한 곳에 모은 모듈.

증강은 두 층으로 나뉜다.

1. 이미지 단위 증강 (RandomResizedCrop, Flip, RandAugment, RandomErasing)
   - 이미지 한 장만 보고 변형하므로 DataLoader의 transform 안에서 처리한다.
   - 라벨은 바뀌지 않는다.
2. 배치 단위 증강 (Mixup, CutMix)
   - 배치 안의 두 이미지를 섞어야 하므로 transform에서는 못 하고 학습 루프에서 처리한다.
   - 라벨도 섞인 비율대로 "soft target"으로 바뀌고, 손실 함수도 거기에 맞춰야 한다.

모든 옵션의 기본값은 "꺼짐"이다. 옵션을 하나도 주지 않으면 이 모듈을 거쳐도
기존 파이프라인과 똑같이 동작한다(tests/test_augment.py에서 확인).

val/test에는 증강을 쓰지 않는다. 그쪽 transform은 data.py에 따로 있고 이 모듈과 무관하다.
"""
import math

from timm.data import Mixup
from timm.data.mixup import mixup_target
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# 증강 옵션의 "꺼짐" 기본값. 이 값이 두 군데에서 쓰인다.
# 1) 각 실험 config.py: 옵션을 안 줬을 때의 기본값
# 2) 각 실험 CHECKPOINT_DEFAULTS와 checkpoint.read_checkpoint: 증강 옵션이 생기기 전에 저장된 옛 checkpoint는 이 키들이 없으므로
#    "이 값으로 학습했다"고 간주하고 비교한다. (그래야 옛 checkpoint를 --mixup 등을 켠 채 이어 학습하는 실수를 막을 수 있다.)
AUGMENT_DEFAULTS = dict(mixup=0.0, cutmix=0.0, random_erasing=0.0, crop_min_scale=0.08)


def validate_augment_args(args):
    """증강 옵션 값이 유효한지 학습 시작 전에 검사한다. NaN/inf도 걸러낸다.

    조건을 `not lo <= v <= hi` 꼴로 쓰는 게 요령이다. NaN은 모든 비교가 거짓이라 이 꼴에서는
    자동으로 걸리고, `v < 0` 같은 꼴로 쓰면 NaN이 그냥 통과해 버린다(나중에 timm 안에서 터진다).
    """
    if not 0 <= args.mixup < math.inf or not 0 <= args.cutmix < math.inf:
        raise ValueError(f"--mixup/--cutmix must be finite numbers >= 0 (0 = off), got {args.mixup}, {args.cutmix}")
    if not 0 <= args.random_erasing <= 1:
        raise ValueError(f"--random-erasing must be a probability in [0, 1] (0 = off), got {args.random_erasing}")
    if not 0 < args.crop_min_scale <= 1:
        raise ValueError(f"--crop-min-scale must be in (0, 1], got {args.crop_min_scale}")


def build_train_transform(image_size, crop_min_scale, random_erasing, norm):
    """이미지 단위 증강을 하나의 transform으로 조립한다.

    crop_min_scale: RandomResizedCrop이 원본에서 잘라낼 최소 면적 비율.
        0.08이 torchvision 기본값(= 기존 동작)이다. 값을 키울수록 crop이 약해진다.
    random_erasing: 이미지 안의 사각형 영역을 지울 확률. 0이면 아예 넣지 않는다.
    norm: 정규화 transform. data.py가 val/test와 같은 것을 쓰도록 밖에서 받는다.
    """
    ops = [
        transforms.RandomResizedCrop(image_size, scale=(crop_min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        norm,
    ]
    if random_erasing > 0:
        # RandomErasing은 텐서에서 동작하므로 ToTensor 뒤에 둔다.
        # 정규화 "뒤"에 두는 이유: value="random"이 표준정규분포 노이즈를 채우는데,
        # 정규화된 입력의 분포와 맞추려면 정규화 이후여야 자연스럽다(DeiT도 같은 순서).
        ops.append(transforms.RandomErasing(p=random_erasing, value="random"))
    return transforms.Compose(ops)


def build_mixup(mixup, cutmix, label_smoothing, *, num_classes):
    """Mixup/CutMix 객체를 만든다. 둘 다 꺼져 있으면 None을 돌려준다.

    mixup, cutmix: 각 기법의 alpha. 클수록 두 이미지를 더 반반에 가깝게 섞는다. 0이면 끈다.
    둘 다 켜면 timm이 배치마다 절반 확률(switch_prob=0.5)로 하나를 골라 적용한다.
    prob=1.0은 "켜져 있으면 모든 배치에 적용"이라는 뜻이다(DeiT 설정).

    label_smoothing을 여기서 같이 받는 이유: timm Mixup은 soft target을 만들 때
    label smoothing까지 한꺼번에 처리한다. 그래서 켜져 있을 때 손실 함수(SoftTargetCrossEntropy)는
    label smoothing을 따로 걸지 않는다(training.py 참고).
    """
    if mixup <= 0 and cutmix <= 0:
        return None
    return Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, prob=1.0, switch_prob=0.5,
                 mode="batch", label_smoothing=label_smoothing, num_classes=num_classes)


def augment_batch(mixup_fn, images, target, label_smoothing):
    """배치에 Mixup/CutMix를 적용하고 (images, target)을 돌려준다.

    mixup_fn이 None이면 아무것도 하지 않고 입력을 그대로 돌려준다(target도 정수 라벨 그대로).
    켜져 있으면 target은 (배치, 클래스 수) 모양의 soft target으로 바뀐다.

    홀수 크기 배치 주의: timm Mixup은 이미지를 절반씩 짝지으므로 배치 크기가 짝수여야 한다.
    마지막 배치가 홀수일 수 있어서(데이터 수가 배치 크기로 나누어떨어지지 않을 때),
    그 배치는 섞지 않고 label smoothing만 적용한 soft target으로 바꿔서 돌려준다.
    이렇게 해야 손실 함수가 항상 같은 형태(soft target)의 입력을 받는다.
    """
    if mixup_fn is None:
        return images, target
    if images.shape[0] % 2 == 1:
        # lam=1.0은 "첫 번째 이미지 100%"라는 뜻이라 섞지 않은 것과 같다.
        # (timm 1.0.x의 mixup_target은 target과 같은 device에 텐서를 만들어 주므로 device 인자가 없다.)
        return images, mixup_target(target, mixup_fn.num_classes, lam=1.0, smoothing=label_smoothing)
    return mixup_fn(images, target)
