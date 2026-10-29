"""Experiment settings exposed through one command-line interface."""
import argparse

import torch

from .augment import AUGMENT_DEFAULTS


def parser():
    p = argparse.ArgumentParser(description="Food-101 scratch ViT training")
    p.add_argument("--experiment", help="Optional label stored with results")
    p.add_argument("--model", default="vit_tiny_patch16_224", help="timm VisionTransformer architecture")
    p.add_argument("--mode", choices=["fp32", "qat"], help="Required for training; restored from --eval-checkpoint for evaluation")
    p.add_argument("--weight-bits", type=int, choices=[2, 3, 4, 8], default=2)
    p.add_argument("--activation-bits", type=int, choices=[2, 3, 4, 8], default=2)
    p.add_argument("--use-cropr", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cropr-pruning-rate", type=int, default=8,
                   help="Tokens removed per Cropr module; original first module removes one extra")
    p.add_argument("--cropr-llf", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--qat-scope", choices=["all", "backbone"], default="all",
                   help="all also quantizes Cropr; backbone leaves Cropr in FP32")
    p.add_argument("--aux-loss-weight", type=float, default=1.0,
                   help="Weight of each Cropr auxiliary loss (original Cropr: 1.0)")
    p.add_argument("--freeze-observers-epoch", type=int, default=-1,
                   help="Zero-based epoch; -1 updates QAT observers throughout training")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    # 데이터 증강 옵션. 기본값은 모두 "기존 동작 그대로"이며 구현은 vitqat/augment.py에 있다.
    p.add_argument("--mixup", type=float, default=AUGMENT_DEFAULTS["mixup"],
                   help="Mixup alpha (0 = off). Standard recipe (DeiT): 0.8")
    p.add_argument("--cutmix", type=float, default=AUGMENT_DEFAULTS["cutmix"],
                   help="CutMix alpha (0 = off). Standard recipe (DeiT): 1.0")
    p.add_argument("--random-erasing", type=float, default=AUGMENT_DEFAULTS["random_erasing"],
                   help="RandomErasing probability (0 = off). Standard recipe (DeiT): 0.25")
    p.add_argument("--crop-min-scale", type=float, default=AUGMENT_DEFAULTS["crop_min_scale"],
                   help="Minimum area fraction of RandomResizedCrop (0.08 = torchvision default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--resume")
    p.add_argument("--eval-checkpoint")
    p.add_argument("--eval-backend", choices=["tensorrt", "pytorch"], default="tensorrt",
                   help="Backend for --eval-checkpoint; TensorRT requires CUDA and FP32/W2A8. Training is unchanged.")
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument("--latency-repeats", type=int, default=100)
    return p
