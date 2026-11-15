"""Epoch/checkpoint lifecycle. Experiments supply model, training, evaluation and reporting."""
import copy
import json
from pathlib import Path
import torch

from .checkpoint import read_checkpoint, save_checkpoint, restore_model
from .reproducibility import seed_all


def run(args, *, build_model, optimizer_factory, train_epoch_fn, evaluate_fn, measure_model, report_results,
        data_factory, loader_factory, selection_score, checkpoint_defaults=None, restore_model=restore_model):
    args = copy.copy(args)
    if args.eval_checkpoint:
        raise ValueError("Use the experiment inference entry point for --eval-checkpoint")
    if args.batch_size < 1 or args.workers < 0 or args.epochs < 1:
        raise ValueError("Invalid batch size, workers or epoch count")
    if args.latency_repeats < 1 or args.latency_warmup < 0:
        raise ValueError("Require latency_repeats >= 1 and latency_warmup >= 0")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (Path(args.resume).resolve() != (output / "last.pt").resolve() or not (output / "best.pt").exists()):
        raise ValueError("Resume must use this output directory's last.pt and retain best.pt")
    if not args.resume and any(output.iterdir()):
        raise ValueError("Choose an empty output directory or explicitly resume")
    seed_all(args.seed)
    train_data, val_data, test_data, manifest = data_factory(args)
    model = build_model(args).to(args.device)
    optimizer = optimizer_factory(model, args)
    start, best = 0, float("-inf")
    if args.resume:
        checkpoint = read_checkpoint(args.resume, model, args, manifest["sha256"], optimizer, config_defaults=checkpoint_defaults, restore_model=restore_model)
        start, best = checkpoint["epoch"] + 1, checkpoint["best_score"]
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
    for epoch in range(start, args.epochs):
        train_loader = loader_factory(train_data, args, training=True, epoch=epoch)
        stats = train_epoch_fn(model, train_loader, optimizer, args, epoch)
        val = evaluate_fn(model, loader_factory(val_data, args), args.device)
        score = selection_score(val)  # Higher is better; a recipe may negate a loss.
        improved = score > best
        best = max(best, score)
        record = dict(epoch=epoch, **stats, **{f"val_{key}": value for key, value in val.items()})
        print(json.dumps(record), flush=True)
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(output / "last.pt", model, optimizer, epoch, best, args, manifest["sha256"])
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, epoch, best, args, manifest["sha256"])
    selected = output / "best.pt"
    checkpoint = read_checkpoint(selected, model, args, manifest["sha256"], config_defaults=checkpoint_defaults, restore_model=restore_model)
    test = evaluate_fn(model, loader_factory(test_data, args), args.device)
    metrics = measure_model(model, args)
    return report_results(model, args, test, checkpoint, selected, manifest, output, metrics)
