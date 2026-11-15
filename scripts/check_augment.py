"""증강이 의도대로 걸리는지 그림으로 확인하는 스크립트 (학습과는 무관).

실행 예 (프로젝트 루트에서):
    python scripts/check_augment.py --data-root data --out augment_check \\
        --mixup 0.8 --cutmix 1.0 --random-erasing 0.25
    python scripts/check_augment.py --image photo1.jpg photo2.jpg ...   # Food-101 없이 임의의 사진으로

만들어지는 파일 (--out 폴더):
    same_image.png  : 같은 원본 1장을 N번 증강한 결과. 윗줄=기본 증강, 아랫줄=지정한 옵션 적용
    mixed_batch.png : 서로 다른 이미지 8장에 Mixup/CutMix를 적용한 결과 (옵션을 줬을 때만)

Food-101이 --data-root 아래에 있어야 한다(없으면 torchvision으로 먼저 받아 둔다).
"""
import argparse
import sys
from pathlib import Path

# `python scripts/check_augment.py`로 실행하면 프로젝트 루트가 import 경로에 없어서 추가해 준다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.augment import AUGMENT_DEFAULTS, augment_batch, build_mixup, build_train_transform

# data.py와 같은 정규화. 저장할 때는 아래 denormalize로 되돌려야 원래 색으로 보인다.
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NORM = transforms.Normalize(MEAN, STD)
LABEL_SMOOTHING = 0.1  # 학습 기본값(--label-smoothing)과 같다


def denormalize(x):
    """정규화된 텐서를 0~1 범위 이미지로 되돌린다 (PNG 저장용)."""
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--image", nargs="+", metavar="FILE",
                   help="Food-101 대신 사용할 이미지 파일들. 라벨이 없으므로 증강이 사진 위에서 "
                        "어떻게 보이는지만 확인할 때 쓴다")
    p.add_argument("--out", default="augment_check")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n", type=int, default=8, help="같은 이미지를 몇 번 증강해 볼지")
    p.add_argument("--mixup", type=float, default=AUGMENT_DEFAULTS["mixup"])
    p.add_argument("--cutmix", type=float, default=AUGMENT_DEFAULTS["cutmix"])
    p.add_argument("--random-erasing", type=float, default=AUGMENT_DEFAULTS["random_erasing"])
    p.add_argument("--crop-min-scale", type=float, default=AUGMENT_DEFAULTS["crop_min_scale"])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 이미지 공급원을 하나로 통일한다: get(i) → (PIL 이미지, 라벨), classes[라벨] → 이름, total → 개수.
    if args.image:
        pool = [Image.open(f).convert("RGB") for f in args.image]
        classes = [Path(f).stem for f in args.image]
        get, total = (lambda i: (pool[i % len(pool)], i % len(pool))), len(pool)
    else:
        # transform=None이면 증강 전 원본(PIL)이 나온다.
        data = datasets.Food101(args.data_root, split="train")
        classes, get, total = data.classes, (lambda i: data[i]), len(data)

    def name(c):
        # soft target의 상위 클래스가 실제 라벨 밖(label smoothing 몫)일 수도 있어서 범위를 확인한다.
        return classes[c] if c < len(classes) else f"class#{c}"

    base_tf = build_train_transform(args.image_size, AUGMENT_DEFAULTS["crop_min_scale"], 0.0, NORM)  # 기존 파이프라인
    aug_tf = build_train_transform(args.image_size, args.crop_min_scale, args.random_erasing, NORM)

    # 1) 같은 이미지를 여러 번 증강
    image, label = get(0)
    rows = [torch.stack([denormalize(tf(image)) for _ in range(args.n)]) for tf in (base_tf, aug_tf)]
    save_image(torch.cat(rows), out / "same_image.png", nrow=args.n)
    print(f"[same_image.png] 클래스={name(label)}  윗줄=기본 증강, 아랫줄=지정 옵션")

    # 2) Mixup/CutMix 배치 확인
    mixup_fn = build_mixup(args.mixup, args.cutmix, LABEL_SMOOTHING, num_classes=101)
    if mixup_fn is None:
        print("[mixed_batch.png] --mixup/--cutmix가 0이라 생략")
        return
    # 이미지가 8장보다 적으면(--image 모드) 돌려 쓴다. 같은 이미지끼리 섞일 수도 있다.
    indices = torch.randperm(total)[:8].tolist() if total >= 8 else list(range(8))
    items = [get(i) for i in indices]
    images = torch.stack([aug_tf(img) for img, _ in items])
    target = torch.tensor([lbl for _, lbl in items])
    mixed, soft = augment_batch(mixup_fn, images.clone(), target, LABEL_SMOOTHING)
    save_image(torch.stack([denormalize(x) for x in mixed]), out / "mixed_batch.png", nrow=4)
    print("[mixed_batch.png] 4열 x 2행. 각 이미지의 soft target 상위 2개 클래스:")
    for i, row in enumerate(soft):
        weight, cls = row.topk(2)
        print(f"  #{i}: " + ", ".join(f"{name(c)}={w:.2f}" for w, c in zip(weight, cls)))


if __name__ == "__main__":
    main()
