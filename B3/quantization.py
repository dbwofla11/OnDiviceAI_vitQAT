"""Static affine fake quantization of the Cropr inference path."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver

from .cropr_adapter import CrossAttention


BIT_WIDTHS = (2, 3, 4, 8)


class Quantizer(FakeQuantize):
    def __init__(self, bits, weight=False):
        options = dict(observer=MinMaxObserver, qscheme=torch.per_tensor_affine)
        if weight:
            options = dict(observer=PerChannelMinMaxObserver, qscheme=torch.per_channel_affine,
                           ch_axis=0)
        super().__init__(quant_min=0, quant_max=2**bits - 1, dtype=torch.quint8, **options)
        self.bits = bits
        self.disable_fake_quant()

    def get_extra_state(self):
        return self.bits

    def set_extra_state(self, bits):
        if bits != self.bits:
            raise ValueError("Checkpoint bit width disagrees with the configured PTQ quantizer")


class PTQLinear(nn.Linear):
    def forward(self, x):
        return self.output_quant(F.linear(self.input_quant(x), self.weight_quant(self.weight), self.bias))


class PTQConv2d(nn.Conv2d):
    def forward(self, x):
        return self.output_quant(self._conv_forward(self.input_quant(x), self.weight_quant(self.weight), self.bias))


class PTQAttention(nn.Module):
    """Quantize both attention matmul operands, retaining the timm parameter names."""
    def __init__(self, original, activation_bits):
        super().__init__()
        for name in ("qkv", "q_norm", "k_norm", "attn_drop", "proj", "proj_drop"):
            setattr(self, name, getattr(original, name))
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.scale = original.scale
        self.q_quant = Quantizer(activation_bits)
        self.k_quant = Quantizer(activation_bits)
        self.v_quant = Quantizer(activation_bits)
        self.prob_quant = Quantizer(activation_bits)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_quant(self.q_norm(q))
        k = self.k_quant(self.k_norm(k))
        v = self.v_quant(v)
        prob = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        prob = self.prob_quant(self.attn_drop(prob))
        x = (prob @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class PTQCrossAttention(CrossAttention):
    """Quantize the learned query and both operands of the inference scorer."""

    def forward_scorer(self, x):
        x = self.attn_norm(x)
        b, n = x.shape[:2]
        queries = self.query_quant(self.queries).expand(b, -1, -1)
        q = self.q_quant(self.q(queries)).reshape(b, self.num_queries, -1, self.head_dim).transpose(1, 2)
        k = self.k_quant(self.k(x)).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        return (q @ k.transpose(-2, -1)).sum((1, 2))


def prepare_ptq(model, weight_bits, activation_bits, scope="all"):
    if weight_bits not in BIT_WIDTHS or activation_bits not in BIT_WIDTHS:
        raise ValueError(f"Supported PTQ bit widths: {BIT_WIDTHS}")
    if scope not in ("all", "backbone"):
        raise ValueError("PTQ scope must be all or backbone")
    if any(isinstance(module, Quantizer) for module in model.modules()):
        raise ValueError("Model already contains PTQ quantizers")
    for block in model.blocks:
        block.attn = PTQAttention(block.attn, activation_bits)
    for name, module in list(model.named_modules()):
        if name.startswith("cropr."):
            if scope == "all" and isinstance(module, CrossAttention):
                module.__class__ = PTQCrossAttention
                module.query_quant = Quantizer(weight_bits, weight=True)
                module.q_quant = Quantizer(activation_bits)
                module.k_quant = Quantizer(activation_bits)
            # The adapter uses identity Q/K projections. All Cropr Linear
            # layers belong to auxiliary branches used only in FP32 training.
            continue
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.__class__ = PTQLinear if isinstance(module, nn.Linear) else PTQConv2d
            module.input_quant = Quantizer(activation_bits)
            module.weight_quant = Quantizer(weight_bits, weight=True)
            module.output_quant = Quantizer(activation_bits)
    return model.eval().requires_grad_(False)


def validate_observers(model):
    for name, quant in model.named_modules():
        if not isinstance(quant, Quantizer):
            continue
        observer = quant.activation_post_process
        if (not quant.fake_quant_enabled.item() or quant.observer_enabled.item()
                or not observer.min_val.numel() or not torch.isfinite(observer.min_val).all()
                or not torch.isfinite(observer.max_val).all()
                or not (observer.min_val <= observer.max_val).all()
                or not torch.isfinite(quant.scale).all() or not (quant.scale > 0).all()
                or not ((quant.zero_point >= 0) & (quant.zero_point <= quant.quant_max)).all()):
            raise ValueError(f"Invalid or uncalibrated PTQ observer at {name}")


def storage_metrics(model):
    """Storage estimate for this experiment's affine fake-quant representation."""
    estimated = sum(p.numel() * p.element_size() for p in model.parameters())
    tensors = []
    for module in model.modules():
        if isinstance(module, (PTQLinear, PTQConv2d)):
            tensors.append((module.weight, module.weight_quant.bits))
        if isinstance(module, CrossAttention) and isinstance(getattr(module, "query_quant", None), Quantizer):
            tensors.append((module.queries, module.query_quant.bits))
    for tensor, bits in tensors:
        estimated += math.ceil(tensor.numel() * bits / 8) - tensor.numel() * tensor.element_size()
        estimated += tensor.shape[0] * 8
    estimated += sum(8 for module in model.modules()
                     if isinstance(module, Quantizer) and module.ch_axis == -1)
    count = sum(tensor.numel() for tensor, _ in tensors)
    return dict(storage_estimated_packed_bytes=estimated, quantized_weight_params=count,
                latency_backend="fake_quant_fp32" if count else "fp32")
