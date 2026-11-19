"""Calibrate a trained FP32 Cropr checkpoint on Food-101 training images."""
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import Subset

from utils.data import loader, make_food101_data
from utils.reproducibility import configure_backend, seed_all
from .config import CHECKPOINT_DEFAULTS, validate_model_config
from .model import build_model, restore_model
from .quantization import BIT_WIDTHS, Quantizer, prepare_ptq, validate_observers


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="FP32 training checkpoint (best.pt)")
    p.add_argument("--output", required=True, help="Directory for ptq.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--calibration-samples", type=int, default=1024)
    p.add_argument("--weight-bits", type=int, choices=BIT_WIDTHS, default=2)
    p.add_argument("--activation-bits", type=int, choices=BIT_WIDTHS, default=8)
    p.add_argument("--ptq-scope", choices=("all", "backbone"), default="all")
    return p


@torch.no_grad()
def calibrate(model, batches, device):
    model.eval()
    quantizers = [m for m in model.modules() if isinstance(m, Quantizer)]
    if not quantizers:
        raise ValueError("Prepare PTQ quantizers before calibration")
    for quant in quantizers:
        observer = quant.activation_post_process
        observer.reset_min_max_vals()
        # PyTorch 2.3 resets per-channel ranges on CPU even for a CUDA model.
        observer.to(quant.scale.device)
        quant.enable_observer()
        quant.disable_fake_quant()
    samples = 0
    for images, _ in batches:
        model(images.to(device))
        samples += len(images)
    if not samples:
        raise ValueError("Empty calibration split")
    for quant in quantizers:
        quant.disable_observer()
        quant.enable_fake_quant()
    validate_observers(model)
    return samples


def main(opts):
    if opts.calibration_samples < 1 or opts.batch_size < 1 or opts.workers < 0:
        raise ValueError("Calibration requires positive sample and batch sizes and workers >= 0")
    output = Path(opts.output) / "ptq.pt"
    if output.resolve() == Path(opts.checkpoint).resolve():
        raise ValueError("Choose a different output directory to preserve the FP32 checkpoint")
    configure_backend(deterministic=True, allow_tf32=False)
    if opts.threads > 0:
        torch.set_num_threads(opts.threads)
    checkpoint = torch.load(opts.checkpoint, map_location="cpu", weights_only=True)
    checkpoint.pop("optimizer", None)
    config = {**CHECKPOINT_DEFAULTS, **checkpoint["config"]}
    validate_model_config(config)
    if config["mode"] != "fp32":
        raise ValueError("PTQ calibration requires an FP32 source checkpoint")
    args = argparse.Namespace(**{**config, **vars(opts), "eval_checkpoint": opts.checkpoint, "resume": None})
    seed_all(args.seed)
    train, val, _, manifest = make_food101_data(args)
    if checkpoint["split_hash"] != manifest["sha256"]:
        raise ValueError("Checkpoint data split does not match")
    # Both subsets index the official train dataset. Use the evaluation
    # transforms with training indices, so calibration sees inference inputs.
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))
    indices = [train.indices[i] for i in order[:opts.calibration_samples].tolist()]
    calibration_data = Subset(val.dataset, indices)
    model = build_model(args)
    restore_model(model, checkpoint)
    prepare_ptq(model, opts.weight_bits, opts.activation_bits, opts.ptq_scope)
    model.to(args.device)
    samples = calibrate(model, loader(calibration_data, args), args.device)

    config.update(mode="ptq", weight_bits=opts.weight_bits, activation_bits=opts.activation_bits,
                  ptq_scope=opts.ptq_scope, experiment="cropr_ptq")
    artifact = dict(model=model.state_dict(), config=config, split_hash=manifest["sha256"],
                    epoch=checkpoint["epoch"], best_score=checkpoint.get("best_score", checkpoint.get("best_acc")),
                    source_checkpoint=str(opts.checkpoint), classes=manifest["classes"],
                    calibration=dict(samples=samples, train_indices=indices))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(artifact, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Calibrated W{opts.weight_bits}A{opts.activation_bits} with {samples} training images -> {output}")
    return output


if __name__ == "__main__":
    main(parser().parse_args())
