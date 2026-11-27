"""INT8 execution building blocks shared by the experiments, for deployment measurements.

Every experiment keeps its own fake quantizers for training and accuracy (A2/A3/A4/B2: affine
FakeQuantize, B1: LSQ/LSQ+). Fake quantization runs in FP32, so its latency is not that of integer
kernels. This module rewrites a copy of a trained model so that every quantized product runs on INT8
operands with the *same* learned grid, for TensorRT (zero-point-0 INT8 Q/DQ):

- An activation grid is x_hat = scale * u + offset with integer codes u in [0, levels]. 8-bit codes are
  shifted to u - 128 so they fit signed INT8; narrower codes fit as they are. The real offset stays in FP32:
  folded into the bias for products with constant weights, or added as row/column-sum corrections for
  products of two activations (attention), so the zero point never has to be integer or zero.
- Weight codes are exact integers held by Q/DQ with scale 1. Per-channel steps multiply the product. Codes
  that do not fit INT8 (8-bit affine weights far from a centered zero point) are shifted by 128, and the
  resulting per-channel offset times the input row sum is added in FP32.
- Codes below 8 bits are clipped before Q, which itself saturates at [-128, 127].

Weights of any bit width use INT8 storage, so TensorRT runs INT8 kernels, not packed 2/4-bit ones. Softmax,
LayerNorm, GELU, residual adds and the offset corrections stay FP32. Exporting a graph does not prove that
TensorRT picks INT8 kernels: the engine's layer report shows the precision it chose.
"""
import copy
import inspect
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

INT8_MIN, INT8_MAX = -128, 127


@dataclass(frozen=True)
class Grid:
    """Per-tensor grid: x_hat = scale * u + offset with integer codes u in [0, levels]."""
    scale: float
    offset: float
    levels: int

    def __post_init__(self):
        if not 0 < self.scale < float("inf") or not -float("inf") < self.offset < float("inf"):
            raise ValueError("A grid needs a positive finite scale and a finite offset")
        if not 0 < self.levels <= 255:
            raise ValueError("INT8 execution supports grids of at most 8 bits")


def affine_grid(fake_quant):
    """Grid of a calibrated per-tensor affine FakeQuantize: x_hat = s * (q - z), q in [quant_min, quant_max]."""
    fake_quant = getattr(fake_quant, "fake_quant", fake_quant)  # Accept the experiments' Quantizer wrapper.
    if fake_quant.scale.numel() != 1:
        raise ValueError("Activation grids are per tensor")
    scale, zero = float(fake_quant.scale.item()), int(fake_quant.zero_point.item())
    low, high = int(fake_quant.quant_min), int(fake_quant.quant_max)
    if not low <= zero <= high:
        raise ValueError("Zero point outside the quantization range")
    return Grid(scale, scale * (low - zero), high - low)


def lsq_grid(step, offset, bits, asymmetric):
    """Grid of an LSQ/LSQ+ quantizer: asymmetric x_hat = s u + beta, symmetric x_hat = s (u - 2^(b-1))."""
    levels = 2 ** bits - 1
    return Grid(step, offset, levels) if asymmetric else Grid(step, -step * 2 ** (bits - 1), levels)


@dataclass
class WeightCodes:
    """w_hat[o] = scale[o] * codes[o] + shift[o] with INT8 codes; shift is None when every code fits INT8."""
    codes: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor = None


def weight_codes(w_hat, scale, zero=None):
    """Integer codes of dequantized weights w_hat = scale_o * (q - zero_o) along output channel 0.

    scale/zero are per output channel or per tensor. The codes are recovered from the fake-quantized values
    themselves, so they are exactly the grid points training used.
    """
    w_hat = w_hat.detach().float()
    channels = w_hat.shape[0]
    scale = scale.detach().float().reshape(-1).expand(channels).clone()
    flat = w_hat.reshape(channels, -1)
    codes = torch.round(flat / scale[:, None])
    shift = None
    if codes.min() < INT8_MIN or codes.max() > INT8_MAX:
        if zero is None:
            raise ValueError("Weight codes do not fit INT8 and no zero point was given")
        zero = zero.detach().float().reshape(-1).expand(channels)
        codes = codes + zero[:, None] - 128  # q - 128
        shift = scale * (128 - zero)
    rebuilt = codes * scale[:, None] + (0 if shift is None else shift[:, None])
    if codes.min() < INT8_MIN or codes.max() > INT8_MAX or not torch.allclose(rebuilt, flat, rtol=1e-5,
                                                                             atol=1e-6 * flat.abs().max().item()):
        raise ValueError("Weights are not on the given grid")
    return WeightCodes(codes.reshape(w_hat.shape), scale, shift)


class QDQ(nn.Module):
    """INT8 fake quantization with zero point 0, exported as QuantizeLinear/DequantizeLinear."""

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("zero", torch.zeros((), dtype=torch.int32))

    def forward(self, x):
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero, INT8_MIN, INT8_MAX)


class ActivationQDQ(nn.Module):
    """x -> scale * code with the grid's real offset removed; x_hat = output + offset."""

    def __init__(self, grid):
        super().__init__()
        shifted = grid.levels > INT8_MAX  # 8-bit codes [0, 255] become [-128, 127].
        self.offset = grid.offset + (128 * grid.scale if shifted else 0.0)
        self.high = None if shifted else grid.levels * grid.scale
        self.qdq = QDQ(grid.scale)

    def forward(self, x):
        z = x - self.offset if self.offset else x
        if self.high is not None:
            z = z.clamp(0.0, self.high)
        return self.qdq(z)


class GridQuant(nn.Module):
    """Fake quantization to a grid through INT8 Q/DQ: returns x_hat (offset added back)."""

    def __init__(self, grid):
        super().__init__()
        self.activation = ActivationQDQ(grid)

    def forward(self, x):
        out = self.activation(x)
        return out + self.activation.offset if self.activation.offset else out


def _folded_bias(bias, weight, offset):
    """bias + offset * sum_k w_hat_k for a constant input offset (LSQ+ Eq. 5); exact in float64."""
    codes = weight.codes.double().reshape(len(weight.scale), -1)
    total = weight.scale.double() * codes.sum(1)
    if weight.shift is not None:
        total = total + codes.shape[1] * weight.shift.double()
    base = torch.zeros(len(weight.scale), dtype=torch.float64) if bias is None else bias.detach().double()
    return (base + offset * total).float()


class Int8Linear(nn.Module):
    """y = x_hat @ w_hat^T + bias on INT8 operands, then the optional output fake quantization."""

    def __init__(self, input_grid, weight, bias=None, output_grid=None):
        super().__init__()
        self.activation = ActivationQDQ(input_grid)
        self.weight_qdq = QDQ(1.0)
        self.register_buffer("codes", weight.codes.float().reshape(len(weight.scale), -1).t().contiguous())
        self.register_buffer("w_scale", weight.scale.float())
        self.register_buffer("w_shift", None if weight.shift is None else weight.shift.float())
        self.register_buffer("bias", _folded_bias(bias, weight, self.activation.offset))
        self.output = None if output_grid is None else GridQuant(output_grid)

    def forward(self, x):
        xq = self.activation(x)
        y = torch.matmul(xq, self.weight_qdq(self.codes)) * self.w_scale + self.bias
        if self.w_shift is not None:
            y = y + xq.sum(-1, keepdim=True) * self.w_shift
        return y if self.output is None else self.output(y)


class Int8Conv2d(nn.Module):
    """Convolution counterpart of Int8Linear (weights [out, in/groups, kh, kw])."""

    def __init__(self, conv, input_grid, weight, output_grid=None):
        super().__init__()
        if conv.groups != 1 or conv.dilation != (1, 1) or tuple(conv.padding) != (0, 0):
            # With padding the zeros would stand for x_hat = offset, so the folded bias would be wrong at borders.
            raise ValueError("INT8 export supports convolutions without padding, groups=1, dilation=1 (patch embedding)")
        self.activation = ActivationQDQ(input_grid)
        self.weight_qdq = QDQ(1.0)
        self.register_buffer("codes", weight.codes.float().reshape(conv.weight.shape).contiguous())
        self.register_buffer("w_scale", weight.scale.float().view(1, -1, 1, 1))
        self.register_buffer("w_shift", None if weight.shift is None else weight.shift.float().view(1, -1, 1, 1))
        self.register_buffer("bias", _folded_bias(conv.bias, weight, self.activation.offset).view(1, -1, 1, 1))
        self.options = dict(stride=conv.stride)
        self.kernel_size = conv.kernel_size
        self.output = None if output_grid is None else GridQuant(output_grid)

    def forward(self, x):
        xq = self.activation(x)
        y = F.conv2d(xq, self.weight_qdq(self.codes), **self.options) * self.w_scale + self.bias
        if self.w_shift is not None:  # Sum of each input window, without a Conv node.
            area = self.kernel_size[0] * self.kernel_size[1]
            window = F.avg_pool2d(xq.sum(1, keepdim=True), self.kernel_size, **self.options) * area
            y = y + window * self.w_shift
        return y if self.output is None else self.output(y)


class Int8MatMul(nn.Module):
    """a_hat @ b_hat for two activations: INT8 product plus FP32 corrections for the real offsets.

    With a_hat = aq + da and b_hat = bq + db: a_hat b_hat = aq bq + db rowsum(aq) + da colsum(bq) + K da db.
    """

    def __init__(self, a_grid, b_grid):
        super().__init__()
        self.a, self.b = ActivationQDQ(a_grid), ActivationQDQ(b_grid)

    def forward(self, a, b):
        aq, bq = self.a(a), self.b(b)
        out = aq @ bq
        da, db = self.a.offset, self.b.offset
        if db:
            out = out + db * aq.sum(-1, keepdim=True)
        if da:
            out = out + da * bq.sum(-2, keepdim=True)
        if da and db:
            out = out + a.shape[-1] * da * db
        return out


class Tap(nn.Identity):
    """Named identity: routers that hook `prob_quant` receive the dequantized probabilities here."""


class Int8Attention(nn.Module):
    """timm attention with quantized Q, K, V and probabilities (the affine QAT experiments' QATAttention).

    QK^T and probability x V run as INT8 products of the trained grids; the dequantized probabilities pass
    through `prob_quant` so CLS-attention routers capture the same tensor as in fake-quant evaluation.
    """

    def __init__(self, attention, qkv, proj):
        super().__init__()
        self.qkv, self.proj = qkv, proj
        for name in ("q_norm", "k_norm", "attn_drop", "proj_drop"):
            setattr(self, name, getattr(attention, name))
        self.num_heads, self.head_dim, self.scale = attention.num_heads, attention.head_dim, attention.scale
        self.qk = Int8MatMul(affine_grid(attention.q_quant), affine_grid(attention.k_quant))
        self.pv = Int8MatMul(affine_grid(attention.prob_quant), affine_grid(attention.v_quant))
        self.probabilities = GridQuant(affine_grid(attention.prob_quant))
        self.prob_quant = Tap()

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        prob = self.attn_drop((self.qk(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        self.prob_quant(self.probabilities(prob))  # Only read by hooks (routers); dead code otherwise.
        x = self.pv(prob, v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


def _affine_parts(module):
    """(input grid, weight codes, output grid) of a fake-quant Linear/Conv of the affine QAT experiments."""
    weight_fq = module.weight_quant.fake_quant
    codes = weight_codes(module.weight_quant(module.weight), weight_fq.scale, weight_fq.zero_point)
    output = getattr(module, "output_quant", None)
    return affine_grid(module.input_quant), codes, None if output is None else affine_grid(output)


def _is_affine_layer(module):
    return (isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(getattr(module, "input_quant", None), "fake_quant")
            and hasattr(getattr(module, "weight_quant", None), "fake_quant"))


def _is_affine_attention(module):
    return all(hasattr(getattr(module, name, None), "fake_quant") for name in ("q_quant", "k_quant", "v_quant", "prob_quant")) \
        and hasattr(module, "qkv") and hasattr(module, "proj")


def int8_layer(module):
    """INT8 form of one affine fake-quant Linear or Conv2d (input, weight and optional output quantizers)."""
    grid, codes, output = _affine_parts(module)
    if isinstance(module, nn.Linear):
        return Int8Linear(grid, codes, module.bias, output)
    return Int8Conv2d(module, grid, codes, output)


def _replace(model, name, replacement):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, replacement)


def convert_affine_qat(model):
    """Copy of an affine fake-quant model (A2/A3/A4/B2) with every quantized product in INT8 form.

    Linear/Conv layers with input/weight(/output) quantizers and attention with Q/K/V/probability quantizers
    are converted; other modules, including pruning wrappers and routers, run unchanged around them.
    """
    model = copy.deepcopy(model).cpu().float().eval().requires_grad_(False)
    for name, module in list(model.named_modules()):
        if _is_affine_attention(module):
            _replace(model, name, Int8Attention(module, int8_layer(module.qkv), int8_layer(module.proj)))
    for name, module in list(model.named_modules()):
        if _is_affine_layer(module):
            _replace(model, name, int8_layer(module))
    return model


def int8_products(model):
    """Number of INT8 matrix products (and convolutions) in a converted model."""
    return sum(1 if isinstance(m, (Int8Linear, Int8Conv2d, Int8MatMul)) else 0 for m in model.modules())


def export_onnx(model, example, path):
    """Fixed-shape ONNX export (input "images", output "logits"); returns the checked graph.

    Constant folding is off so the graph is the plain trace; the Q/DQ of constant weights is what makes a
    product INT8, and audit_int8_products counts them.
    """
    import onnx

    options = dict(opset_version=17, input_names=["images"], output_names=["logits"], do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False  # Newer PyTorch versions default to the dynamo exporter.
    with torch.no_grad():
        torch.onnx.export(model.cpu().eval(), example.cpu(), str(path), **options)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return graph


def audit_int8_products(graph):
    """MatMul/Conv nodes and whether both inputs come from DequantizeLinear (needed for INT8 kernels).

    This checks the exported graph, not TensorRT kernel selection.
    """
    producers = {output: node for node in graph.graph.node for output in node.output}
    rows = []
    for node in graph.graph.node:
        if node.op_type in ("MatMul", "Conv"):
            dq = [producers.get(name) is not None and producers[name].op_type == "DequantizeLinear"
                  for name in node.input[:2]]
            rows.append(dict(name=node.name, op=node.op_type, int8=all(dq)))
    return rows
