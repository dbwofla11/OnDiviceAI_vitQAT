"""A4-owned CLS/Cropr pruning API."""
from .inference import InferencePrunedViT, PruningConfig, AttentionRouter
from .model import PrunedOutput, attach_pruning

__all__ = ["attach_pruning", "PrunedOutput", "PruningConfig",
           "InferencePrunedViT", "AttentionRouter"]
