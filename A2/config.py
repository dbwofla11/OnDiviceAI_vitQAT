"""A2 experiment settings and checkpoint compatibility checks."""
import argparse
from utils.cli import parser as common_parser
from utils.augment import AUGMENT_DEFAULTS, validate_augment_args

# Only backward-compatible, non-structural defaults may be inferred.
CHECKPOINT_REQUIRED = {'model', 'mode', 'use_cropr', 'image_size', 'drop_path', 'weight_bits', 'activation_bits', 'qat_scope'}
CHECKPOINT_DEFAULTS = dict(AUGMENT_DEFAULTS,
                          pretrained=False, cls_pruning=None, cls_pruning_rate=16,
                          cropr_pruning_rate=8, cropr_llf=False, aux_loss_weight=1.0,
                          freeze_observers_epoch=-1)


def parser(*, inference=False):
    p = common_parser(__doc__)
    p.add_argument("--eval-checkpoint", required=inference)
    p.set_defaults(resume=None)
    if inference:
        return p
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=AUGMENT_DEFAULTS["mixup"])
    p.add_argument("--cutmix", type=float, default=AUGMENT_DEFAULTS["cutmix"])
    p.add_argument("--random-erasing", type=float, default=AUGMENT_DEFAULTS["random_erasing"])
    p.add_argument("--crop-min-scale", type=float, default=AUGMENT_DEFAULTS["crop_min_scale"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--model", default="vit_tiny_patch16_224")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False,
                   help="Initialize a new training run from timm pretrained weights; never redownload on resume/evaluation")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--mode", choices=['qat'], default='qat')
    # Retain historical checkpoint keys without exposing unrelated method switches.
    p.set_defaults(use_cropr=False, cls_pruning=None, cls_pruning_rate=16,
                   cropr_pruning_rate=8, cropr_llf=False, weight_bits=2, activation_bits=2,
                   qat_scope="all", aux_loss_weight=1.0, freeze_observers_epoch=-1)
    p.add_argument("--weight-bits", type=int, choices=[2, 3, 4, 8], default=2)
    p.add_argument("--activation-bits", type=int, choices=[2, 3, 4, 8], default=2)
    p.add_argument("--freeze-observers-epoch", type=int, default=-1)
    return p


def validate(args):
    missing = CHECKPOINT_REQUIRED - vars(args).keys()
    if missing:
        raise ValueError(f"Missing model config keys: {sorted(missing)}")
    if args.mode != 'qat' or args.use_cropr != False or args.cls_pruning != None:
        raise ValueError("A2 requires mode='qat', use_cropr=False, cls_pruning=None")
    if args.image_size <= 0 or not 0 <= args.drop_path < 1:
        raise ValueError("Invalid image size or drop path")
    if not args.eval_checkpoint and (args.freeze_observers_epoch == 0 or args.freeze_observers_epoch < -1):
        raise ValueError("Observers must run for at least one epoch before freezing")
    if not args.eval_checkpoint:
        if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
            raise ValueError("Require epochs >= 1 and 0 <= warmup_epochs < epochs")
        if args.accum_steps < 1:
            raise ValueError("Invalid accumulation steps")
        if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or args.clip_grad <= 0:
            raise ValueError("Require 0 <= min_lr <= lr, lr > 0, clip_grad > 0")
        if not 0 <= args.label_smoothing < 1 or args.weight_decay < 0:
            raise ValueError("Invalid label smoothing or weight decay")
        validate_augment_args(args)
