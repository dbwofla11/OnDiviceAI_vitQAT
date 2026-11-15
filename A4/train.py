"""A4 training recipe: model, optimizer, forward/loss and observer policy."""
from utils.data import make_food101_data as make_data, loader
from utils.reproducibility import configure_backend
import copy
import math
from torch import nn
from timm.loss import SoftTargetCrossEntropy
from utils.augment import augment_batch, build_mixup
from utils.constants import NUM_CLASSES
import torch
from utils.optim import weight_decay_groups
from utils.runner import run
from .config import parser, validate, CHECKPOINT_DEFAULTS
from .model import build_model, restore_model
from .metrics import measure_model, report_results
from .inference import evaluate
from .quantization import freeze_observers


def create_optimizer(model, args):
    return torch.optim.AdamW(weight_decay_groups(model, args.weight_decay), lr=args.lr)


def compute_loss(model, images, target, criterion):
    logits = model(images)
    loss = criterion(logits, target)
    return loss, {"main": loss, "aux": loss.new_zeros(())}


def learning_rate(step, total, warmup, peak, minimum):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def train_epoch(model, batches, optimizer, args, epoch, steps_per_epoch=None):
    """This experiment selects augmentation, criterion, LR schedule and gradient clipping."""
    if steps_per_epoch is None:
        steps_per_epoch = math.ceil(len(batches) / args.accum_steps)
    freeze_observers(model, args.freeze_observers_epoch >= 0 and epoch >= args.freeze_observers_epoch)
    model.train()
    mixup_fn = build_mixup(args.mixup, args.cutmix, args.label_smoothing, num_classes=NUM_CLASSES)
    criterion = (nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) if mixup_fn is None
                 else SoftTargetCrossEntropy())
    optimizer.zero_grad(set_to_none=True)
    totals, total_loss, count, updates = {}, 0.0, 0, 0
    for i, (images, target) in enumerate(batches):
        images, target = images.to(args.device), target.to(args.device)
        images, target = augment_batch(mixup_fn, images, target, args.label_smoothing)
        start = (i // args.accum_steps) * args.accum_steps
        end = min(start + args.accum_steps, len(batches))
        group_samples = min(end * batches.batch_size, len(batches.dataset)) - start * batches.batch_size
        if i == start:
            lr = learning_rate(epoch * steps_per_epoch + updates, args.epochs * steps_per_epoch,
                             args.warmup_epochs * steps_per_epoch, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
        loss, components = compute_loss(model, images, target, criterion)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {i}")
        (loss * images.shape[0] / group_samples).backward()
        if i + 1 == end:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_loss += loss.item() * images.shape[0]
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * images.shape[0]
        count += images.shape[0]
    return dict(train_loss=total_loss / count, lr=lr, optimizer_steps=updates,
                **{f"train_{key}_loss": value / count for key, value in totals.items()})


def selection_score(validation):
    return validation["top1_acc"]


def main(args):
    configure_backend(deterministic=True, allow_tf32=False)
    if args.eval_checkpoint:
        from .inference import main as infer
        return infer(args)
    args = copy.copy(args)
    validate(args)
    if args.experiment is None:
        args.experiment = 'cls_train_and_inference_qat'
    return run(args, build_model=build_model, optimizer_factory=create_optimizer,
               train_epoch_fn=train_epoch, evaluate_fn=evaluate,
               measure_model=measure_model, report_results=report_results,
               data_factory=make_data, loader_factory=loader, selection_score=selection_score,
               checkpoint_defaults=CHECKPOINT_DEFAULTS, restore_model=restore_model)


if __name__ == "__main__":
    main(parser().parse_args())
