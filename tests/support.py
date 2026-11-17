"""Adapters for historical multi-method regression tests. Not imported by production code."""
import importlib
from .legacy_config import parser
from utils.augment import AUGMENT_DEFAULTS
from utils.checkpoint import read_checkpoint as load_checkpoint
from utils.data import make_food101_data as make_data, loader
from utils.classification import evaluate
from A1.train import create_optimizer
from utils.runner import run as run_training
from utils.evaluation import run as run_evaluation


def experiment(args):
    if args.mode not in ("fp32", "qat"):
        raise ValueError("--mode is required")
    if getattr(args, "cls_pruning", None):
        return "A4"
    if args.use_cropr:
        return "B2"  # FP32 Cropr is B2's backbone before QAT (see build_model).
    return "A2" if args.mode == "qat" else "A1"


def build_model(args):
    if args.use_cropr and args.mode == "fp32" and not getattr(args, "cls_pruning", None):
        # B1 became Cropr + Linear QAT, so the FP32 Cropr reference is B2's model before QAT.
        return importlib.import_module("B2.model").build_backbone(args)
    return importlib.import_module(experiment(args) + ".model").build_model(args)


def main(args):
    return importlib.import_module(experiment(args) + ".train").main(args)


def train_epoch(model, batches, optimizer, args, epoch, steps_per_epoch=None):
    return importlib.import_module(experiment(args) + ".train").train_epoch(
        model, batches, optimizer, args, epoch, steps_per_epoch)


def measure_model(model, args):
    return importlib.import_module(experiment(args) + ".metrics").measure_model(model, args)


def measure_vit(model, image_size, device, warmup=10, repeats=100):
    quantized = any(type(module).__module__.endswith(".quantization") for module in model.modules())
    args = parser().parse_args(["--mode", "qat" if quantized else "fp32", "--output", "unused"])
    args.use_cropr = getattr(model, "use_cropr", False)
    args.image_size, args.device = image_size, device
    args.latency_warmup, args.latency_repeats = warmup, repeats
    return measure_model(model, args)


def report_results(model, args, test, checkpoint, selected, manifest, output, metrics):
    return importlib.import_module(experiment(args) + ".metrics").report_results(
        model, args, test, checkpoint, selected, manifest, output, metrics)


def run(args, *, build_model, measure_model, train_epoch_fn=train_epoch, evaluate_fn=evaluate,
        data_factory=make_data, optimizer_factory=create_optimizer):
    if args.eval_checkpoint:
        return run_evaluation(args, build_model=build_model, measure_model=measure_model, report_results=report_results,
                              evaluate_fn=evaluate_fn, data_factory=data_factory, loader_factory=loader,
                              validate_config=lambda config: None, config_defaults={**AUGMENT_DEFAULTS, "pretrained": False})
    importlib.import_module(experiment(args) + ".config").validate(args)
    return run_training(args, build_model=build_model, measure_model=measure_model, report_results=report_results,
                        optimizer_factory=optimizer_factory, train_epoch_fn=train_epoch_fn,
                        evaluate_fn=evaluate_fn, data_factory=data_factory, loader_factory=loader,
                        checkpoint_defaults={**AUGMENT_DEFAULTS, "pretrained": False},
                        selection_score=lambda val: val["top1_acc"])


def read_checkpoint(path, model, args, split_hash, optimizer=None):
    return load_checkpoint(path, model, args, split_hash, optimizer,
                           config_defaults={**AUGMENT_DEFAULTS, "pretrained": False})
