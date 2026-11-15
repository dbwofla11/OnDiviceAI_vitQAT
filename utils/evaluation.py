"""Checkpoint restoration and evaluation I/O; experiment callbacks own inference behavior."""
import copy
import json
from pathlib import Path
import torch

from .checkpoint import restore_model
from .reproducibility import seed_all


def run(args, *, build_model, evaluate_fn, measure_model, report_results, validate_config,
        data_factory, loader_factory, config_defaults=None, restore_model=restore_model):
    args = copy.copy(args)
    if not args.eval_checkpoint or args.resume:
        raise ValueError("Evaluation requires --eval-checkpoint without --resume")
    checkpoint = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=True)
    runtime = {"eval_checkpoint", "resume", "output", "data_root", "download", "device", "workers",
               "threads", "batch_size", "latency_warmup", "latency_repeats", "experiment"}
    for key, value in {**(config_defaults or {}), **checkpoint["config"]}.items():
        if key not in runtime:
            setattr(args, key, value)
    if args.experiment is None:
        args.experiment = checkpoint["config"].get("experiment")
    validate_config(args)
    if args.batch_size < 1 or args.workers < 0 or args.latency_repeats < 1 or args.latency_warmup < 0:
        raise ValueError("Invalid evaluation runtime settings")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    seed_all(args.seed)
    _, _, test_data, manifest = data_factory(args)
    if checkpoint["split_hash"] != manifest["sha256"]:
        raise ValueError("Checkpoint data split does not match")
    model = build_model(args).to(args.device)
    restore_model(model, checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
    test = evaluate_fn(model, loader_factory(test_data, args), args.device)
    metrics = measure_model(model, args)
    return report_results(model, args, test, checkpoint, Path(args.eval_checkpoint), manifest, output, metrics)
