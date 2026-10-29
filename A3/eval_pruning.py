"""Paired dense / inference-Cropr evaluation using inherited training settings."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import timm
import torch
import torchvision

from vitqat.augment import AUGMENT_DEFAULTS
from vitqat.data import loader, make_data
from .pruning import attach_pruning
from .pruning.checkpoint import checkpoint_model, pruning_config
from .metrics import measure_vit
from vitqat.models import build_model
from vitqat.quantization import Quantizer, freeze_observers
from vitqat.training import evaluate, read_checkpoint, seed_all


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Upstream QAT-only best.pt")
    p.add_argument("--output", required=True, help="New empty evaluation directory")
    p.add_argument("--evaluation-id", default="S-QAT-P")
    p.add_argument("--data-root", help="Override data location only, not its split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--eval-batch-size", type=int, help="Optional, applies equally to both conditions")
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument("--latency-repeats", type=int, default=100)
    p.add_argument("--pruning-schedule", choices=["cropr-r16-first17-llf"],
                   default="cropr-r16-first17-llf")
    return p


def inherited_args(checkpoint, options):
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("QAT checkpoint must include its original config")
    required = {"model", "mode", "use_cropr", "image_size", "drop_path", "seed",
                "val_fraction", "batch_size", "weight_bits", "activation_bits", "qat_scope",
                "cropr_llf", "cropr_pruning_rate", "data_root", "workers"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing inherited config keys: {sorted(missing)}")
    if config["mode"] != "qat" or config["use_cropr"]:
        raise ValueError("Require a QAT-only checkpoint trained without Cropr")
    if any(config.get(key, False) for key in ("pretrained", "amp", "tf32")):
        raise ValueError("This repository loader supports scratch, no-AMP/no-TF32 checkpoints only")
    args = argparse.Namespace(**{**AUGMENT_DEFAULTS, "threads": 0, **config})
    args.device = options.device
    args.output = options.output
    args.resume, args.eval_checkpoint, args.download = None, options.checkpoint, False
    if options.data_root is not None:
        args.data_root = options.data_root
    if options.workers is not None:
        args.workers = options.workers
    if options.threads is not None:
        args.threads = options.threads
    args.latency_warmup, args.latency_repeats = options.latency_warmup, options.latency_repeats
    if (args.batch_size < 1 or args.workers < 0 or args.threads < 0
            or options.latency_warmup < 0 or options.latency_repeats < 1
            or (options.eval_batch_size is not None and options.eval_batch_size < 1)):
        raise ValueError("Invalid evaluation batch size, workers, threads or latency repetitions")
    return args


def validate_qat_state(model):
    quantizers = [m for m in model.modules() if isinstance(m, Quantizer)]
    if not quantizers:
        raise ValueError("QAT checkpoint must contain quantizers")
    for module in quantizers:
        fq = module.fake_quant
        observer = fq.activation_post_process
        if (observer.min_val.numel() == 0 or observer.max_val.numel() == 0
                or not torch.isfinite(observer.min_val).all()
                or not torch.isfinite(observer.max_val).all()
                or not torch.isfinite(fq.scale).all() or not (fq.scale > 0).all()
                or not fq.fake_quant_enabled.all()):
            raise ValueError("Require trained/calibrated observers and enabled fake quantization")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Non-finite backbone checkpoint state")


def state_digest(model):
    """Byte-exact parameter/buffer digest, without retaining another full model copy."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(options, *, model_factory=build_model, data_factory=make_data):
    output = Path(options.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new empty evaluation output directory")
    backbone_hash = sha256_file(options.checkpoint)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    args = inherited_args(checkpoint, options)
    seed_all(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    _, val_data, test_data, manifest = data_factory(args)
    if manifest["sha256"] != checkpoint.get("split_hash"):
        raise ValueError("Checkpoint data split does not match")
    model = model_factory(args).to(args.device)
    read_checkpoint(options.checkpoint, checkpoint_model(model, args), args, manifest["sha256"])
    validate_qat_state(model)
    model.eval()
    freeze_observers(model, True)
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.fake_quant.disable_observer()
    model.requires_grad_(False)
    pruning = pruning_config(args)
    pruned = attach_pruning(model, config=pruning, apply_during="inference")
    original_state = state_digest(model)
    if len(val_data) == 0:
        raise ValueError("Validation data is required for the latency sample")
    sample_index = args.seed % len(val_data)
    sample = val_data[sample_index][0].unsqueeze(0).to(args.device)
    if options.eval_batch_size is not None:
        args.batch_size = options.eval_batch_size  # Only after strict training-config validation.
    common = dict(
        checkpoint_sha256=backbone_hash, split_hash=manifest["sha256"],
        evaluation_id=options.evaluation_id, model=args.model, mode=args.mode,
        weight_bits=args.weight_bits, activation_bits=args.activation_bits, seed=args.seed,
        image_size=args.image_size, eval_batch_size=args.batch_size, device=args.device,
        device_name=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else platform.processor(),
        torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
        timm_version=timm.__version__, python_version=platform.python_version(),
        cpu_threads=torch.get_num_threads(), amp=False, tf32=False, training_pruning=False,
        latency_sample=dict(split="validation", subset_index=sample_index,
                            dataset_index=int(val_data.indices[sample_index]) if hasattr(val_data, "indices") else sample_index),
    )
    results = {}
    for condition, evaluated_model in (("dense", model), ("pruned", pruned)):
        test = evaluate(evaluated_model, loader(test_data, args), args.device)
        measured = measure_vit(evaluated_model, args.image_size, args.device,
                               args.latency_warmup, args.latency_repeats, sample=sample)
        results[condition] = dict(common, **measured, top1_acc=test["top1_acc"],
                                  test_loss=test["loss"], test_samples=test["samples"],
                                  inference_pruning=condition == "pruned")
        if condition == "pruned":
            results[condition].update(pruning.describe(len(model.blocks), model.patch_embed.num_patches))
    if state_digest(model) != original_state:
        raise RuntimeError("Evaluation changed backbone, quantizer or router state")
    if sha256_file(options.checkpoint) != backbone_hash:
        raise RuntimeError("An input checkpoint changed during evaluation")
    dense, reduced = results["dense"], results["pruned"]
    comparison = dict(checkpoint_sha256=backbone_hash, split_hash=manifest["sha256"],
                      top1_loss_pp=dense["top1_acc"] - reduced["top1_acc"],
                      latency_speedup=dense["latency_ms_median"] / reduced["latency_ms_median"],
                      gflops_ratio=reduced["gflops"] / dense["gflops"],
                      backbone_state_unchanged=True)
    # Guard against replacing inputs or outputs created while evaluation was running.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Evaluation output directory is no longer empty")
    documents = {"evaluation_config": dict(training_config=checkpoint["config"],
                    evaluation_options=vars(options), effective_config=vars(args),
                    checkpoint_sha256=backbone_hash,
                    pruning=pruning.describe(len(model.blocks), model.patch_embed.num_patches)),
                 "split": manifest, "dense_metrics": dense, "pruned_metrics": reduced,
                 "comparison": comparison}
    for name, content in documents.items():
        (output / f"{name}.json").write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [dict(condition=key, **value) for key, value in results.items()]
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return comparison


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
