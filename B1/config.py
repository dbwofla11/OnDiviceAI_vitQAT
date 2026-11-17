"""B1 experiment settings and checkpoint compatibility checks."""
import argparse
from pathlib import Path
from utils.cli import parser as common_parser
from utils.augment import AUGMENT_DEFAULTS, validate_augment_args
from .quantizers import BITS

WEIGHT_QUANTIZERS = ("lsq+", "lsq", "statsq")
ACTIVATION_QUANTIZERS = ("lsq+", "lsq")
# Only backward-compatible, non-structural defaults may be inferred.
CHECKPOINT_REQUIRED = {'model', 'mode', 'use_cropr', 'image_size', 'drop_path', 'cropr_pruning_rate', 'cropr_llf',
                       'weight_bits', 'activation_bits', 'first_last_bits', 'cropr_scorer_bits',
                       'weight_quantizer', 'activation_quantizer'}
CHECKPOINT_DEFAULTS = dict(AUGMENT_DEFAULTS,
                          pretrained=False, init_checkpoint=None, cls_pruning=None)


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
    p.add_argument("--init-checkpoint",
                   help="FP32 checkpoint (best.pt/last.pt/model_state.pt) loaded before QAT is attached; new runs only")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--mode", choices=['linear_qat'], default='linear_qat')
    # Retain the historical checkpoint key without exposing an unrelated method switch.
    p.set_defaults(cls_pruning=None)
    p.add_argument("--weight-bits", type=int, choices=BITS, default=2,
                   help="Weights of Linear layers inside transformer blocks")
    p.add_argument("--activation-bits", type=int, choices=BITS, default=2,
                   help="Inputs of Linear layers inside transformer blocks")
    p.add_argument("--first-last-bits", type=int, choices=[*BITS, 32], default=8,
                   help="Patch embedding and classifier head weight/input bits; 32 keeps them FP32")
    p.add_argument("--cropr-scorer-bits", type=int, choices=[*BITS, 32], default=8,
                   help="Cropr scorer queries/token bits; 32 keeps the scorer FP32")
    p.add_argument("--weight-quantizer", choices=WEIGHT_QUANTIZERS, default="lsq+",
                   help="lsq+: LSQ step with LSQ+ init; lsq: LSQ init; statsq: OFQ statistics-based")
    p.add_argument("--activation-quantizer", choices=ACTIVATION_QUANTIZERS, default="lsq+",
                   help="lsq+: asymmetric with learnable offset; lsq: signed symmetric")
    p.add_argument("--use-cropr", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-use-cropr keeps the same Linear QAT without token pruning (ablation)")
    p.add_argument("--cropr-pruning-rate", type=int, default=8,
                   help="Tokens removed per Cropr module; original first module removes one extra")
    p.add_argument("--cropr-llf", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--aux-loss-weight", type=float, default=1.0,
                   help="Weight of each Cropr auxiliary loss (original Cropr: 1.0)")
    return p


def validate(args):
    missing = CHECKPOINT_REQUIRED - vars(args).keys()
    if missing:
        raise ValueError(f"Missing model config keys: {sorted(missing)}")
    if args.mode != 'linear_qat' or args.cls_pruning != None:
        raise ValueError("B1 requires mode='linear_qat', cls_pruning=None")
    if args.image_size <= 0 or not 0 <= args.drop_path < 1:
        raise ValueError("Invalid image size or drop path")
    if (args.weight_bits not in BITS or args.activation_bits not in BITS
            or args.first_last_bits not in (*BITS, 32) or args.cropr_scorer_bits not in (*BITS, 32)):
        raise ValueError(f"Supported bit widths: {BITS}; first/last and scorer also accept 32 (FP32)")
    if args.weight_quantizer not in WEIGHT_QUANTIZERS or args.activation_quantizer not in ACTIVATION_QUANTIZERS:
        raise ValueError("Unsupported weight or activation quantizer")
    if args.aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be nonnegative")
    if args.use_cropr and args.aux_loss_weight == 0:
        raise ValueError("Cropr learnable queries require a positive auxiliary loss weight")
    if not args.eval_checkpoint:
        if args.pretrained and args.init_checkpoint:
            raise ValueError("Choose either --pretrained or --init-checkpoint")
        if args.init_checkpoint and not args.resume and not Path(args.init_checkpoint).is_file():
            raise ValueError(f"--init-checkpoint not found: {args.init_checkpoint}")
        if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
            raise ValueError("Require epochs >= 1 and 0 <= warmup_epochs < epochs")
        if args.accum_steps < 1:
            raise ValueError("Invalid accumulation steps")
        if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or args.clip_grad <= 0:
            raise ValueError("Require 0 <= min_lr <= lr, lr > 0, clip_grad > 0")
        if not 0 <= args.label_smoothing < 1 or args.weight_decay < 0:
            raise ValueError("Invalid label smoothing or weight decay")
        validate_augment_args(args)
