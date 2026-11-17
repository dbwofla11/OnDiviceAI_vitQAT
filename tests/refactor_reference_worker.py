"""Subprocess worker: execute either the frozen original tree or the current tree."""
import argparse
import importlib
import math
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tree")
    p.add_argument("version", choices=("before", "after"))
    p.add_argument("experiment")
    p.add_argument("output")
    p.add_argument("--align-training-rng", action="store_true")
    p.add_argument("--source-checkpoint")
    options = p.parse_args()
    sys.path.insert(0, options.tree)
    import torch
    from torch.utils.data import TensorDataset

    torch.set_num_threads(2)
    name = options.experiment
    if options.version == "before":
        from vitqat import train as entry
        from vitqat.data import loader
        from vitqat.training import seed_all, rng_state
        build = entry.build_pruned_model
        evaluate = importlib.import_module("vitqat.training").evaluate
    else:
        entry = importlib.import_module(f"{name if name != 'A3' else 'A2'}.train")
        from utils.data import loader
        from utils.reproducibility import seed_all, rng_state, configure_backend
        configure_backend(deterministic=True, allow_tf32=False)
        build, evaluate = entry.build_model, entry.evaluate
    qat = name in {"A2", "A3", "A4", "B2"}
    size = 224 if name == "A3" else 64
    flags = ["--output", options.output, "--device", "cpu", "--workers", "0", "--threads", "2",
             "--image-size", str(size), "--batch-size", "2", "--accum-steps", "2", "--seed", "42",
             "--epochs", "2", "--warmup-epochs", "0", "--drop-path", "0.1", "--mixup", "0.8"]
    if options.version == "before":
        flags += ["--mode", "qat" if qat else "fp32"]
        if name == "B2":
            flags += ["--use-cropr"]
        if name == "A4":
            flags += ["--cls-pruning", "train_and_inference"]
    if name == "B2":
        flags += ["--cropr-pruning-rate", "1"]
    if name == "A4":
        flags += ["--cls-pruning-rate", "1"]
    if qat:
        flags += ["--freeze-observers-epoch", "1"]
    if name == "A3":
        flags += ["--activation-bits", "8"]
    args = entry.parser().parse_args(flags)
    images = torch.randn(3, 3, size, size, generator=torch.Generator().manual_seed(123))
    dataset = TensorDataset(images, torch.tensor([0, 1, 2]))
    seed_all(args.seed)
    model = build(args)
    output = Path(options.output)
    output.mkdir(parents=True)
    initial_rng = rng_state()
    # Save only parameters: QAT buffers initially contain expected infinities.
    torch.save(dict(model={key: value.detach() for key, value in model.named_parameters()},
                    rng=initial_rng), output / "initial.pt")

    if name == "A3":
        from A3.pruning import attach_pruning, PruningConfig
        quant = importlib.import_module("vitqat.quantization" if options.version == "before" else "A3.quantization")
        if options.version == "after":
            from A3.model import build_model, restore_model
            args.eval_checkpoint = "synthetic-upstream.pt"
            # Use the actual A3 restore path on a legacy-format A2 checkpoint.
            checkpoint = torch.load(options.source_checkpoint, weights_only=True)
            model = build_model(args)
            restore_model(model, checkpoint)
        else:
            with torch.no_grad():
                model(images)
            torch.save(dict(model=model.state_dict(), config=vars(args)), options.source_checkpoint)
        model.eval()
        quant.freeze_observers(model, True)
        pruned = attach_pruning(model, config=PruningConfig(), apply_during="inference")
        pruned.eval()
        values = {}
        with torch.no_grad():
            for condition, candidate in (("dense", model), ("pruned", pruned)):
                traces = []
                backbone = getattr(candidate, "backbone", candidate)
                hooks = [block.register_forward_pre_hook(lambda module, inputs: traces.append(inputs[0].shape[1]))
                         for block in backbone.blocks]
                logits = candidate(images)
                for hook in hooks:
                    hook.remove()
                values[condition] = dict(logits=logits, tokens=traces,
                                         metrics=evaluate(candidate, loader(dataset, args), "cpu"))
        torch.save(values, output / "result.pt")
        return

    if options.version == "before":
        # Original runner's optimizer grouping, copied verbatim in meaning (no new utility).
        decay, no_decay = [], []
        excluded = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
        for key, parameter in model.named_parameters():
            (no_decay if parameter.ndim == 1 or key in excluded else decay).append(parameter)
        optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=args.weight_decay),
                                       dict(params=no_decay, weight_decay=0.0)], lr=args.lr)
    else:
        optimizer = entry.create_optimizer(model, args)
    if options.align_training_rng:
        seed_all(73)
    records = []
    for epoch in range(2):
        batches = loader(dataset, args, training=True, epoch=epoch)
        stats = entry.train_epoch(model, batches, optimizer, args, epoch, math.ceil(len(batches) / args.accum_steps))
        metrics = evaluate(model, loader(dataset, args), "cpu")
        with torch.no_grad():
            logits = model(images)
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]
        records.append(dict(stats=stats, metrics=metrics, logits=logits))
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), records=records,
                    rng=rng_state()), output / "result.pt")


if __name__ == "__main__":
    main()
