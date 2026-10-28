"""CLS attention token pruning with the Cropr removal schedule.

Use the Cropr removal schedule and last-layer fusion, with CLS attention Top-K scoring.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PruningConfig:
    pruning_rate: int = 16
    llf: bool = True
    first_stage_extra: int = 1

    def schedule(self, depth, patch_tokens):
        if (type(self.pruning_rate) is not int or self.pruning_rate < 1
                or self.llf is not True or self.first_stage_extra != 1):
            raise ValueError("Require positive integer R, LLF on and first-stage extra=1")
        if depth < 3:
            raise ValueError("LLF requires at least three transformer blocks")
        rates = (self.pruning_rate + 1,) + (self.pruning_rate,) * (depth - 3)
        if sum(rates) >= patch_tokens:
            raise ValueError("Pruning schedule must retain at least one patch token")
        return rates

    def describe(self, depth, patch_tokens):
        rates = self.schedule(depth, patch_tokens)
        return dict(asdict(self), schedule_id=f"cropr-r{self.pruning_rate}-first{rates[0]}-llf",
                    after_blocks=list(range(1, len(rates) + 1)), remove_counts=list(rates),
                    scorer="cls-attention-head-mean",
                    selection="argsort-descending-stable-false",
                    fusion_order="keep,removed_stage1,...,removed_stageN")


class AttentionRouter(nn.Module):
    """Keep CLS and the patches with the largest mean CLS attention."""

    def __init__(self, pruning_rate):
        super().__init__()
        self.pruning_rate = pruning_rate

    def forward(self, x, attention):
        # Fixed-shape export runs after eager validation; do not trace Python guards.
        if not torch.onnx.is_in_onnx_export():
            if attention.ndim != 4 or attention.shape[0] != x.shape[0] or attention.shape[-2:] != (x.shape[1], x.shape[1]):
                raise ValueError("Attention must have shape [B, heads, tokens, tokens]")
            if x.shape[1] <= self.pruning_rate + 1:
                raise ValueError("Router must retain CLS and at least one patch")
        scores = attention.detach()[:, :, 0, :].mean(dim=1).clone()
        scores[:, 0] = torch.inf
        # The default is stable=False; omitting it supports legacy ONNX export.
        order = torch.argsort(scores, dim=1, descending=True)
        reordered = x.gather(1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        keep = x.shape[1] - self.pruning_rate
        return reordered[:, :keep], reordered[:, keep:]


class InferencePrunedViT(nn.Module):
    """Wrap an already-loaded dense ViT without changing backbone state.

    The wrapper shares backbone parameters; it never constructs a new backbone.
    Trace IDs are optional diagnostics and are not calculated during timing.
    """

    def __init__(self, backbone, config=None):
        super().__init__()
        if getattr(backbone, "use_cropr", False) or backbone.num_prefix_tokens != 1:
            raise ValueError("Require a dense backbone with exactly one CLS token")
        for method in ("forward_embed", "forward_head"):
            if not callable(getattr(backbone, method, None)):
                raise ValueError("Backbone must use the repository Cropr adapter forward contract")
        if any(p.dtype != torch.float32 for p in backbone.parameters()):
            raise ValueError("Inference pruning currently supports FP32/fake-quant FP32 backbones")
        self.config = config or PruningConfig()
        rates = self.config.schedule(len(backbone.blocks), backbone.patch_embed.num_patches)
        self.backbone = backbone
        self.routers = nn.ModuleList([AttentionRouter(rate) for rate in rates])
        self.routers.to(next(backbone.parameters()).device)
        self.enabled = True
        self.capture_trace = False
        self.last_trace = None
        self.eval()

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
        if self.training or self.backbone.training:
            raise RuntimeError("InferencePrunedViT cannot train a backbone")
        return self._forward_pruned(images)

    def _forward_pruned(self, images):
        self.last_trace = None
        if not self.enabled:
            return self.backbone(images)
        x = self.backbone.forward_embed(images)
        removed = []
        if self.capture_trace:
            ids = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
            removed_ids, counts = [], []
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                x = torch.cat([x] + removed, dim=1)
                if self.capture_trace:
                    ids = torch.cat([ids] + removed_ids, dim=1)
            if self.capture_trace:
                counts.append(x.shape[1])
            attention = []
            handle = None
            if i < len(self.routers):
                attn = block.attn
                if getattr(attn, "fused_attn", False):
                    raise ValueError("CLS attention pruning requires unfused attention")
                capture = getattr(attn, "prob_quant", attn.attn_drop)
                handle = capture.register_forward_hook(lambda module, inputs, output: attention.append(output))
            try:
                x = block(x)
            finally:
                if handle is not None:
                    handle.remove()
            if i < len(self.routers):
                if len(attention) != 1:
                    raise RuntimeError("Could not capture attention probabilities")
                router = self.routers[i]
                if self.capture_trace:
                    scores = attention[0].detach()[:, :, 0, :].mean(dim=1).clone()
                    scores[:, 0] = torch.inf
                    order = torch.argsort(scores, dim=1, descending=True, stable=False)
                    reordered_ids = ids.gather(1, order)
                    keep = x.shape[1] - router.pruning_rate
                    ids, dropped_ids = reordered_ids[:, :keep], reordered_ids[:, keep:]
                    removed_ids.append(dropped_ids)
                x, dropped = router(x, attention[0])
                removed.append(dropped)
        if self.capture_trace:
            self.last_trace = dict(tokens_per_block=counts, restored_ids=ids.detach().cpu(),
                                   removed_ids=[v.detach().cpu() for v in removed_ids])
        return self.backbone.forward_head(self.backbone.norm(x))
