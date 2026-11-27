"""Attach Linear-layer QAT to the original Cropr ViT without editing cropr/.

Quantized (weight per output channel, input activation per tensor):
- every nn.Linear in the transformer blocks (qkv, attention proj, MLP fc1/fc2): W{w}A{a};
- patch embedding (Conv2d = Linear over 16x16 patches) and classifier head: first_last_bits;
- each Cropr scorer, a linear map x -> x Q^T (learnable queries Q): cropr_scorer_bits.
Left in FP32: attention QK^T/AV matmuls, softmax, LayerNorm, GELU, residual adds, embeddings,
and the training-only Cropr aggregator/auxiliary heads, which are removed at inference.
Training and inference run the same quantizer modules, so both use the same symmetric weight grid
and the same (asymmetric LSQ+ or symmetric LSQ) activation grid. These modules are fake quantization
in FP32; integer.py runs the same grids with integer kernels to measure latency.
"""
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .cropr_adapter import CrossAttention
from .quantizers import ActivationQuantizer, WeightQuantizer, make_activation_quantizer, make_weight_quantizer

FULL_PRECISION = 32


class QuantLinear(nn.Linear):
    def forward(self, x):
        return F.linear(self.input_quant(x), self.weight_quant(self.weight), self.bias)


class QuantConv2d(nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(self.input_quant(x), self.weight_quant(self.weight), self.bias)


class QuantCrossAttention(CrossAttention):
    """Original Cropr cross-attention with a quantized scorer (queries as weight, tokens as input).

    Training (forward) and inference (forward_scorer) share the same quantizers, so tokens are
    ranked with the same quantized scores in both paths. The aggregator branch stays FP32.
    """

    def _query_key(self, x):
        b, n = x.shape[:2]
        queries = self.query_quant(self.queries[0]).unsqueeze(0).expand(b, -1, -1)
        q = self.q(queries).reshape(b, self.num_queries, -1, self.head_dim).transpose(1, 2)
        k = self.k(self.input_quant(x)).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        return q, k

    def forward_scorer(self, x):
        q, k = self._query_key(self.attn_norm(x))
        return (q @ k.transpose(-2, -1)).sum((1, 2))

    def forward(self, x):
        b, n, d = x.shape
        x = self.attn_norm(x)
        q, k = self._query_key(x)
        v = self.v(x).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        attn_sm = (attn * self.scale).softmax(dim=-1)
        x = (attn_sm @ v).transpose(1, 2).reshape(b, self.num_queries, d).squeeze(1)
        x = self.proj(x)
        if hasattr(self, "mlp"):
            x = x + self.mlp(self.mlp_norm(x))
        return x, attn.sum((1, 2))


def _quantize_layer(module, weight_bits, activation_bits, weight_method, activation_method):
    # Swap the class in place: parameter names, values and RNG state are unchanged.
    module.__class__ = QuantLinear if isinstance(module, nn.Linear) else QuantConv2d
    module.weight_quant = make_weight_quantizer(module.weight, weight_bits, weight_method)
    module.input_quant = make_activation_quantizer(activation_bits, activation_method)


def prepare_linear_qat(model, weight_bits, activation_bits, first_last_bits=8, cropr_scorer_bits=8,
                       weight_quantizer="lsq+", activation_quantizer="lsq+"):
    """Convert in place and return the model. Bits of 32 keep that group in FP32."""
    if quantizers(model):
        raise ValueError("Model already contains Linear QAT quantizers")
    # StatsQ targets the low-bit block weights (OFQ); its 2 mean|w| clip is unsuitable for 8-bit
    # layers, so the first/last layers and Cropr scorers use the LSQ+ step size in that case.
    edge_method = "lsq+" if weight_quantizer == "statsq" else weight_quantizer
    if first_last_bits != FULL_PRECISION:
        for module in (model.patch_embed.proj, model.head):
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                raise ValueError("Expected Conv2d patch embedding and Linear classifier head")
            _quantize_layer(module, first_last_bits, first_last_bits, edge_method, activation_quantizer)
    for module in list(model.blocks.modules()):  # list(): converting adds submodules
        if type(module) is nn.Linear:
            _quantize_layer(module, weight_bits, activation_bits, weight_quantizer, activation_quantizer)
    if getattr(model, "use_cropr", False) and cropr_scorer_bits != FULL_PRECISION:
        for pruner in model.cropr:
            attention = pruner.cross_attn
            if not (isinstance(attention.q, nn.Identity) and isinstance(attention.k, nn.Identity)):
                raise ValueError("Scorer quantization assumes the original Identity q/k projections")
            attention.__class__ = QuantCrossAttention
            attention.query_quant = make_weight_quantizer(attention.queries[0], cropr_scorer_bits, edge_method)
            attention.input_quant = make_activation_quantizer(cropr_scorer_bits, activation_quantizer)
    return model


def quantized_modules(model):
    """(module name, quantized weight tensor, weight quantizer) for every converted layer."""
    found = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantConv2d)):
            found.append((name, module.weight, module.weight_quant))
        elif isinstance(module, QuantCrossAttention):
            found.append((name, module.queries, module.query_quant))
    return found


def quantizers(model):
    return [m for m in model.modules() if isinstance(m, (WeightQuantizer, ActivationQuantizer))]


def weight_quantizers(model):
    """(weight tensor exactly as its quantizer sees it, weight quantizer) for every converted layer."""
    found = []
    for module in model.modules():
        if isinstance(module, (QuantLinear, QuantConv2d)):
            found.append((module.weight, module.weight_quant))
        elif isinstance(module, QuantCrossAttention):
            found.append((module.queries[0], module.query_quant))  # One step per query.
    return found


def project_quantizer_steps(model):
    """Run after every optimizer update: keep each learned step inside a usable range.

    With AdamW a step moves by about lr per update, whatever its size. An 8-bit weight step starts near
    4e-4 and an 8-bit activation step can start as small, so at lr 5e-4 a few updates can push them below
    zero and positive() would map them to EPS: the channel, or the whole activation tensor, then
    quantizes to (almost) a constant. The floors only touch such steps; healthy steps are unchanged.
    """
    for weight, quantizer in weight_quantizers(model):
        quantizer.project_step(weight)
    for quantizer in quantizers(model):
        if isinstance(quantizer, ActivationQuantizer):
            quantizer.project_step()


def collapsed_quantizer_channels(model):
    """Weight channels and activation quantizers whose step collapsed (counts are zero after projection).

    A checkpoint trained without project_quantizer_steps may contain channels that positive() clamps to
    EPS: they quantize to (almost) zero instead of their bit width.
    """
    count = sum(quantizer.collapsed_channels(weight) for weight, quantizer in weight_quantizers(model))
    return count + sum(int(q.collapsed()) for q in quantizers(model) if isinstance(q, ActivationQuantizer))


def validate_linear_qat_state(model):
    """Reject restored states that evaluation would otherwise use with untrained ranges."""
    found = quantizers(model)
    if not found:
        raise ValueError("Linear QAT checkpoint must contain quantizers")
    if not all(bool(q.initialized) for q in found if isinstance(q, ActivationQuantizer)):
        raise ValueError("Require activation quantizers initialized by a training batch")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Non-finite Linear QAT checkpoint state")
    collapsed = collapsed_quantizer_channels(model)
    if collapsed:
        # Older checkpoints may hold such channels: evaluate them, but say so (metrics also record the count).
        warnings.warn(f"{collapsed} quantizer channels have a collapsed step and quantize to (almost) zero; "
                      "retrain with the current B1 code", stacklevel=2)
