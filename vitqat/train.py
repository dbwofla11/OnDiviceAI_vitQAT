"""Single entry point: python -m vitqat.train --help."""
from A3.metrics import measure_vit as a3_measure_vit
from A3.pruning import PruningConfig, attach_pruning
from .cli import parser
from .metrics import measure_vit
from .models import build_model
from .quantization import freeze_observers
from .runner import run
from .training import train_epoch as classification_train_epoch


def train_epoch(model, batches, optimizer, args, epoch, steps_per_epoch):
    if args.mode == "qat":
        freeze_observers(model, args.freeze_observers_epoch >= 0 and epoch >= args.freeze_observers_epoch)
    return classification_train_epoch(model, batches, optimizer, args, epoch, steps_per_epoch)


def build_pruned_model(args):
    model = build_model(args)
    if not args.cls_pruning:
        return model
    backbone = model
    model = attach_pruning(backbone, config=PruningConfig(pruning_rate=args.cls_pruning_rate),
                           apply_during=args.cls_pruning)
    # The wrapper hides timm's no_weight_decay; without this pos_embed/cls_token get decayed
    # and the run no longer matches the QAT-only baseline recipe.
    model.no_weight_decay = lambda: {f"backbone.{name}" for name in backbone.no_weight_decay()}
    return model


def measure_model(model, args):
    measure = a3_measure_vit if args.cls_pruning else measure_vit
    metrics = measure(model, args.image_size, args.device, args.latency_warmup, args.latency_repeats)
    metrics.update(cls_pruning=args.cls_pruning, cls_pruning_rate=args.cls_pruning_rate if args.cls_pruning else 0)
    metrics.update(model=args.model, mode=args.mode, use_cropr=args.use_cropr,
                   cropr_pruning_rate=args.cropr_pruning_rate if args.use_cropr else 0,
                   cropr_llf=args.cropr_llf if args.use_cropr else False, qat_scope=args.qat_scope,
                   weight_bits=args.weight_bits if args.mode == "qat" else 32,
                   activation_bits=args.activation_bits if args.mode == "qat" else 32)
    return metrics


def main(args):
    if args.aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be nonnegative")
    if args.use_cropr and args.aux_loss_weight == 0:
        raise ValueError("Cropr learnable queries require a positive auxiliary loss weight")
    if args.mode == "qat" and (args.freeze_observers_epoch == 0 or args.freeze_observers_epoch < -1):
        raise ValueError("Observers must run for at least one epoch before freezing")
    if args.cls_pruning and args.use_cropr:
        raise ValueError("--cls-pruning and --use-cropr are different pruning methods; choose one")
    if args.experiment is None:
        args.experiment = (f"cls_{args.cls_pruning}_{args.mode}" if args.cls_pruning else
                           ("cropr_" + args.mode) if args.use_cropr else ("1-1" if args.mode == "fp32" else "1-2"))
    return run(args, build_model=build_pruned_model, measure_model=measure_model, train_epoch_fn=train_epoch)


if __name__ == "__main__":
    main(parser().parse_args())
