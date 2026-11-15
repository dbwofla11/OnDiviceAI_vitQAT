"""Compatibility imports; new integrations should import .pruning."""
from .pruning import InferencePrunedViT, PruningConfig, AttentionRouter

__all__ = ["InferencePrunedViT", "PruningConfig", "AttentionRouter"]
