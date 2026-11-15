"""utils/augment.py 단위 테스트.

확인하는 것:
1. 옵션을 안 주면 기존 동작과 똑같다 (증강이 "선택형"이라는 약속).
2. Mixup/CutMix가 켜졌을 때 target이 올바른 soft target인지.
3. 홀수 크기 배치(마지막 배치)가 에러 없이 처리되는지.
4. 실제 학습 루프(train_epoch)에 연결했을 때 끝까지 도는지.
5. checkpoint 호환, resume 시 난수 재현, make_data(worker 사용) 연결.
"""
import copy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from timm.loss import SoftTargetCrossEntropy

from utils.constants import NUM_CLASSES
from utils.augment import augment_batch, build_mixup, build_train_transform
from tests.support import parser
from utils.data import loader, make_food101_data as make_data
from tests.support import read_checkpoint
from utils.checkpoint import save_checkpoint
from utils.reproducibility import seed_all
from tests.support import train_epoch

AUGMENT_KEYS = ["mixup", "cutmix", "random_erasing", "crop_min_scale"]

NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SMOOTHING = 0.1


@pytest.fixture(autouse=True)
def fixed_seed():
    # timm Mixup은 numpy 전역 난수를 쓰므로 torch와 numpy 둘 다 고정한다.
    torch.manual_seed(0)
    np.random.seed(0)


def test_everything_off_is_a_noop():
    """옵션이 모두 꺼져 있으면 Mixup 객체도 없고, 입력과 target이 그대로 돌아온다."""
    assert build_mixup(0.0, 0.0, SMOOTHING, num_classes=101) is None
    images, target = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])
    out_images, out_target = augment_batch(None, images, target, SMOOTHING)
    assert out_images is images and out_target is target


def test_default_transform_matches_previous_pipeline():
    """기본값 transform은 기존 구성(Crop, Flip, RandAugment, ToTensor, Normalize)과 같다."""
    ops = build_train_transform(32, 0.08, 0.0, NORM).transforms
    assert [type(op).__name__ for op in ops] == [
        "RandomResizedCrop", "RandomHorizontalFlip", "RandAugment", "ToTensor", "Normalize"]
    assert ops[0].scale == (0.08, 1.0)


def test_random_erasing_changes_pixels():
    """p=1이면 매번 어딘가가 지워지므로, 같은 입력에서 지우지 않은 결과와 달라야 한다."""
    image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    erased = build_train_transform(32, 1.0, 1.0, NORM).transforms
    plain = erased[:-1]  # 마지막(RandomErasing)만 뺀 같은 파이프라인
    differing = 0
    for _ in range(20):
        seed = int(torch.randint(0, 10**6, (1,)))
        torch.manual_seed(seed)
        a = transforms.Compose(erased)(image)
        torch.manual_seed(seed)
        b = transforms.Compose(plain)(image)
        assert a.shape == (3, 32, 32) and torch.isfinite(a).all()
        differing += int(not torch.equal(a, b))
    assert differing > 0


@pytest.mark.parametrize("mixup,cutmix", [(0.8, 0.0), (0.0, 1.0), (0.8, 1.0)])
def test_soft_target_rows_sum_to_one(mixup, cutmix):
    mixup_fn = build_mixup(mixup, cutmix, SMOOTHING, num_classes=101)
    images, target = torch.randn(8, 3, 16, 16), torch.randint(0, NUM_CLASSES, (8,))
    out_images, soft = augment_batch(mixup_fn, images.clone(), target, SMOOTHING)
    assert out_images.shape == images.shape
    assert soft.shape == (8, NUM_CLASSES)
    assert torch.allclose(soft.sum(dim=1), torch.ones(8), atol=1e-5)
    assert not torch.equal(out_images, images)  # 실제로 섞였다


def test_odd_batch_is_not_mixed_but_still_soft():
    """홀수 배치: 이미지는 그대로, target은 label smoothing만 적용된 soft target."""
    mixup_fn = build_mixup(0.8, 1.0, SMOOTHING, num_classes=101)
    images, target = torch.randn(5, 3, 16, 16), torch.randint(0, NUM_CLASSES, (5,))
    out_images, soft = augment_batch(mixup_fn, images.clone(), target, SMOOTHING)
    assert torch.equal(out_images, images)
    assert soft.shape == (5, NUM_CLASSES)
    assert torch.allclose(soft.sum(dim=1), torch.ones(5), atol=1e-5)


def test_soft_loss_equals_cross_entropy_when_not_mixed():
    """섞지 않은 경우(홀수 배치) soft 손실 값이 기존 label-smoothing CrossEntropy와 같아야 한다.
    이 둘이 어긋나면 증강을 켠 실험과 끈 실험의 loss를 비교할 수 없게 된다."""
    mixup_fn = build_mixup(0.8, 0.0, SMOOTHING, num_classes=101)
    logits, target = torch.randn(5, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (5,))
    _, soft = augment_batch(mixup_fn, torch.randn(5, 3, 8, 8), target, SMOOTHING)
    expected = nn.CrossEntropyLoss(label_smoothing=SMOOTHING)(logits, target)
    assert torch.allclose(SoftTargetCrossEntropy()(logits, soft), expected, atol=1e-5)


def test_train_epoch_runs_with_mixup_and_odd_last_batch(tmp_path):
    """학습 루프 통합 확인. 데이터 5개, 배치 2개씩이면 마지막 배치가 홀수(1개)가 된다."""
    args = parser().parse_args(["--mode", "fp32", "--output", str(tmp_path), "--mixup", "0.8"])
    args.device, args.epochs, args.warmup_epochs = "cpu", 2, 0
    model = nn.Linear(4, NUM_CLASSES)
    data = TensorDataset(torch.randn(5, 4), torch.randint(0, NUM_CLASSES, (5,)))
    stats = train_epoch(model, DataLoader(data, batch_size=2), torch.optim.SGD(model.parameters(), lr=0.1),
                        args, 0, 3)
    assert np.isfinite(stats["train_loss"])


# ── 아래는 checkpoint / resume / 데이터 로더 연결 확인 ─────────────────────────

def small_args(tmp_path, **overrides):
    """CPU에서 빨리 도는 최소 설정. 증강 옵션은 overrides로 켠다."""
    args = parser().parse_args(["--mode", "fp32", "--output", str(tmp_path / "out")])
    args.device, args.image_size, args.epochs, args.warmup_epochs = "cpu", 32, 2, 0
    args.workers, args.batch_size, args.drop_path = 0, 2, 0.0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def linear_setup():
    """Mixup 난수가 학습 결과에 미치는 영향만 보려고 모델은 가장 작은 Linear를 쓴다.
    데이터 5개/배치 2개 → 배치는 2, 2, 1개. 앞의 두 배치는 섞이고 마지막 홀수 배치는 안 섞인다."""
    model = nn.Linear(4, NUM_CLASSES)
    data = TensorDataset(torch.randn(5, 4), torch.randint(0, NUM_CLASSES, (5,)))
    return model, torch.optim.SGD(model.parameters(), lr=0.1), DataLoader(data, batch_size=2)


def test_checkpoint_saved_before_augment_options_loads_only_with_augmentation_off(tmp_path):
    """증강 옵션이 생기기 전에 저장된 옛 checkpoint는 "증강 없이 학습한 것"으로 취급한다.
    - 증강을 끈 채 이어 학습하면 통과한다.
    - 증강을 켠 채 이어 학습하려 하면 거부한다 (앞쪽 epoch와 뒤쪽 epoch의 증강이 달라지는 실험 오염 방지)."""
    args = small_args(tmp_path)
    model, optimizer, _ = linear_setup()
    path = tmp_path / "old.pt"
    save_checkpoint(path, model, optimizer, 0, 1.0, args, "split")
    # 옛 checkpoint 흉내: config에서 새 옵션 키를 지우고 다시 저장한다.
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in AUGMENT_KEYS:
        del state["config"][key]
    torch.save(state, path)
    read_checkpoint(path, model, args, "split", optimizer)  # 증강을 끈 채로는 에러 없이 통과해야 한다
    for key, value in [("mixup", 0.8), ("cutmix", 1.0), ("random_erasing", 0.25), ("crop_min_scale", 0.5)]:
        with pytest.raises(ValueError, match=key):
            read_checkpoint(path, model, small_args(tmp_path, **{key: value}), "split")


def test_resume_with_different_augment_setting_is_rejected(tmp_path):
    """Mixup을 켜고 저장한 checkpoint를 Mixup 없이 이어 학습하면 실험이 오염되므로 막혀야 한다."""
    model, optimizer, _ = linear_setup()
    path = tmp_path / "mixup.pt"
    save_checkpoint(path, model, optimizer, 0, 1.0, small_args(tmp_path, mixup=0.8), "split")
    with pytest.raises(ValueError, match="mixup"):
        read_checkpoint(path, model, small_args(tmp_path, mixup=0.0), "split")


def test_resume_reproduces_mixup_randomness(tmp_path):
    """학습을 중간에 끊고 checkpoint에서 이어도, 쉬지 않고 학습한 것과 같은 결과가 나와야 한다.
    Mixup의 난수(numpy 전역 RNG)가 checkpoint에 저장·복원되는지가 핵심이다."""
    args = small_args(tmp_path, mixup=0.8)
    seed_all(5)
    model, optimizer, batches = linear_setup()
    train_epoch(model, batches, optimizer, args, 0, 3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0, 1.0, args, "split")
    train_epoch(model, batches, optimizer, args, 1, 3)  # 쉬지 않고 이어서 학습한 기준 결과
    expected = copy.deepcopy(model.state_dict())

    def resumed(perturb_rng):
        other = nn.Linear(4, NUM_CLASSES)
        other_optimizer = torch.optim.SGD(other.parameters(), lr=0.1)
        read_checkpoint(path, other, args, "split", other_optimizer)  # 여기서 RNG도 복원된다
        if perturb_rng:
            np.random.seed(999)  # 대조군: 복원된 난수를 일부러 망가뜨린다
        train_epoch(other, batches, other_optimizer, args, 1, 3)
        return other.state_dict()

    good, bad = resumed(False), resumed(True)
    assert all(torch.equal(value, good[key]) for key, value in expected.items())
    # 대조군이 달라져야 위 비교가 "난수 복원 덕분에 같은 것"임을 믿을 수 있다.
    assert any(not torch.equal(value, bad[key]) for key, value in expected.items())


def test_make_data_applies_augment_to_train_only_even_with_workers(food_fixture):
    """make_data → DataLoader(worker 사용) 경로에서 증강이 train에만 걸리고 batch가 정상 생성되는지."""
    args = small_args(food_fixture, data_root=str(food_fixture), workers=2,
                      random_erasing=1.0, crop_min_scale=0.5)
    train_data, val_data, _, _ = make_data(args)
    names = lambda subset: [type(op).__name__ for op in subset.dataset.transform.transforms]
    assert "RandomErasing" in names(train_data)
    assert "RandomErasing" not in names(val_data)
    assert train_data.dataset.transform.transforms[0].scale == (0.5, 1.0)
    images, labels = next(iter(loader(train_data, args, training=True)))
    assert images.shape == (2, 3, 32, 32) and torch.isfinite(images).all()


# ── 옵션 값 검증과 클래스 수 단일 출처 ─────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("mixup", float("nan")), ("cutmix", float("inf")), ("random_erasing", 1.5), ("crop_min_scale", 0.0),
])
def test_invalid_augment_option_is_rejected_with_field_name(tmp_path, field, value):
    """잘못된 값은 학습 시작 전에, 어떤 옵션이 문제인지 알 수 있는 메시지로 거부해야 한다.
    run()을 거쳐 검증이 실제로 연결돼 있는지도 함께 확인한다 (검증은 run() 맨 앞에서 일어난다)."""
    from tests.support import run
    args = small_args(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        run(args, build_model=None, measure_model=None)
