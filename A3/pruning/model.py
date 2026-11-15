"""Training/inference adapters around an existing repository ViT backbone."""
from dataclasses import dataclass

import torch
from torch import nn

from .inference import InferencePrunedViT, PruningConfig


@dataclass
class PrunedOutput:
    logits: torch.Tensor
    auxiliary_logits: tuple

    def auxiliary_loss(self, target, criterion):
        """Use the caller's criterion, including soft-target Mixup loss if needed."""
        return sum((criterion(pred, target) for pred in self.auxiliary_logits),
                   self.logits.new_zeros(()))


class InferenceModePruner(InferencePrunedViT):
    """Dense during train(), CLS attention pruning during eval()."""

    def __init__(self, backbone, config):
        was_training = backbone.training
        super().__init__(backbone, config)
        self.train(was_training)

    def forward(self, images):
        if self.training:
            self.last_trace = None
            return self.backbone(images)
        return super().forward(images)


class TrainAndInferencePruner(InferencePrunedViT):
    """Use CLS attention Top-K in both train and eval, returning logits."""

    def __init__(self, backbone, config):
        was_training = backbone.training
        super().__init__(backbone, config)
        self.train(was_training)

    def forward(self, images):
        return self._forward_pruned(images)


class TrainOnlyPruner(TrainAndInferencePruner):
    """Prune during training; run the dense backbone during evaluation."""

    def forward(self, images):
        if not self.training:
            self.last_trace = None
            return self.backbone(images)
        return self._forward_pruned(images)


class TrainablePruner(nn.Module):
    """Cropr with auxiliary learning, FP32 routers and unchanged backbone modules.

    Training returns PrunedOutput; evaluation returns logits. Create the optimizer
    AFTER attachment so queries and auxiliary heads are included.
    """

    def __init__(self, backbone, config, queries=None):
        # Legacy learned-query API only; CLS experiment paths do not import Cropr.
        from cropr.cls.cropr import Cropr

        super().__init__()
        # Reuse the validation without random fallback in the inference API.
        rates = config.schedule(len(backbone.blocks), backbone.patch_embed.num_patches)
        if queries is not None:
            expected = {str(i + 1) for i in range(len(rates))}
            if set(queries) != expected or any(q.shape != (1, 1, backbone.embed_dim) for q in queries.values()):
                raise ValueError("Invalid Cropr query mapping or dimension")
        if getattr(backbone, "use_cropr", False) or backbone.num_prefix_tokens != 1:
            raise ValueError("Require a dense backbone with one CLS token")
        if any(p.dtype != torch.float32 for p in backbone.parameters()):
            raise ValueError("Require FP32/fake-quant FP32 backbone")
        self.backbone, self.config = backbone, config
        self.enabled = True
        self.cropr = nn.ModuleList([
            Cropr(pruning_rate=rate, embed_dim=backbone.embed_dim,
                  num_classes=backbone.num_classes, num_queries=1, num_heads=1,
                  pre_attn_norm=False, q_proj=False, k_proj=False, v_proj=False,
                  mlp=True, mlp_ratio=4, training=True) for rate in rates
        ]).to(next(backbone.parameters()).device)
        if queries is not None:
            for i, stage in enumerate(self.cropr):
                stage.cross_attn.queries.data.copy_(queries[str(i + 1)])
        self.train(backbone.training)

    @property
    def blocks(self):
        return self.backbone.blocks

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    @property
    def num_prefix_tokens(self):
        return self.backbone.num_prefix_tokens

    def forward(self, images):
        if not self.enabled:
            logits = self.backbone(images)
            return PrunedOutput(logits, ()) if self.training else logits
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        dropped, predictions = [], []
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                x = torch.cat([x] + dropped, dim=1)
            x = block(x)
            if i < len(self.cropr):
                x, removed, pred = self.cropr[i](x, inference=not self.training)
                dropped.append(removed)
                if pred is not None:
                    predictions.append(pred)
        logits = self.backbone.forward_head(self.backbone.norm(x))
        return PrunedOutput(logits, tuple(predictions)) if self.training else logits

    def export_queries(self):
        """Tensor export only; caller must supply truthful training provenance."""
        return {str(i + 1): stage.cross_attn.queries.detach().cpu().clone()
                for i, stage in enumerate(self.cropr)}


def attach_pruning(model, *, queries=None, config=None, pruning_rate=16,
                   llf=True, apply_during="inference"):
    """Attach without editing the original class or reinitializing its weights.

    inference: train=dense, eval=CLS attention Top-K pruning; no queries.
    train: train=CLS attention Top-K pruning, eval=dense; no queries.
    train_and_inference: CLS attention Top-K in both train and eval; no queries.
    cropr_train_and_inference: legacy learned-query Cropr and auxiliary heads.
    """
    if isinstance(model, (InferencePrunedViT, TrainablePruner)):
        raise ValueError("Pruning is already attached")
    config = config or PruningConfig(pruning_rate=pruning_rate, llf=llf)
    if apply_during == "inference":
        if queries is not None:
            raise ValueError("CLS attention pruning does not use queries")
        return InferenceModePruner(model, config)
    if apply_during == "train_and_inference":
        if queries is not None:
            raise ValueError("CLS attention pruning does not use queries")
        return TrainAndInferencePruner(model, config)
    if apply_during == "train":
        if queries is not None:
            raise ValueError("CLS attention pruning does not use queries")
        return TrainOnlyPruner(model, config)
    if apply_during == "cropr_train_and_inference":
        return TrainablePruner(model, config, queries)
    raise ValueError("apply_during must be train, inference, train_and_inference or cropr_train_and_inference")
