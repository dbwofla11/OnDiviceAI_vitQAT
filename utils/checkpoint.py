"""Checkpoint/state I/O; compatibility defaults are supplied by the experiment."""
import torch
from .reproducibility import rng_state, restore_rng


def save_checkpoint(path, model, optimizer, epoch, best, args, split_hash):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                 best_score=best, config=vars(args), split_hash=split_hash, rng=rng_state())
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_model(model, checkpoint):
    """Default state restoration for models without an experiment-specific format."""
    model.load_state_dict(checkpoint["model"], strict=True)


def read_checkpoint(path, model, args, split_hash, optimizer=None, *, config_defaults=None,
                    restore_model=restore_model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "best_score" not in checkpoint and "best_acc" in checkpoint:
        checkpoint["best_score"] = checkpoint["best_acc"]
    ignored = {"resume", "eval_checkpoint", "output", "data_root", "download", "device", "workers",
               "latency_warmup", "latency_repeats", "threads"}
    saved_config = {**(config_defaults or {}), **checkpoint["config"]}
    for key, value in saved_config.items():
        # Retired CLI options in older checkpoints do not affect supported settings.
        if key in vars(args) and key not in ignored and getattr(args, key) != value:
            raise ValueError(f"Checkpoint config mismatch: {key}: {value!r} != {getattr(args, key)!r}")
    if checkpoint["split_hash"] != split_hash:
        raise ValueError("Checkpoint data split does not match")
    restore_model(model, checkpoint)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
    return checkpoint
