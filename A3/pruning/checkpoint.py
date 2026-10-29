"""Restore the parameter namespace and execution mode of CLS-pruned training."""
from . import PruningConfig, attach_pruning


def pruning_config(args):
    return PruningConfig(pruning_rate=getattr(args, "cls_pruning_rate", 16))


def checkpoint_model(backbone, args):
    mode = getattr(args, "cls_pruning", None)
    if not mode:
        return backbone
    return attach_pruning(backbone, config=pruning_config(args), apply_during=mode)
