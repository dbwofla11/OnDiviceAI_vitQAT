"""Export a trained B1 model as an INT8 Q/DQ ONNX graph for TensorRT, keeping the learned grids.

TensorRT only has zero-point-0 INT8 Q/DQ, which does not have a place for the real LSQ+ offset. The
removed symmetric remapping changed the trained activation grid; here the grid stays as trained:

- Activation codes are u in [0, 2^b - 1] (LSQ+) or q in [-2^(b-1), 2^(b-1) - 1] (LSQ). Q/DQ with scale s
  and zero point 0 stores them as signed INT8. For 8-bit LSQ+ the codes are shifted by 128
  (u - 128 in [-128, 127]), so x_hat = s * (u - 128) + (beta + 128 s); narrower codes fit INT8 directly.
  The real offset (beta, or beta + 128 s) is folded into the layer bias, as LSQ+ Eq. 5 does.
- Weight codes are exact integers held by Q/DQ with scale 1. Per-channel steps multiply the product,
  so no per-channel Q/DQ is required for the matrix product.
- Values below b bits are clipped before Q: Q itself saturates at [-128, 127].

Weights of any bit width are stored in INT8 (2-bit codes are in [-2, 1]), so this measures INT8 kernels.
Attention QK^T/AV, softmax, LayerNorm, GELU, residual adds and biases stay FP32 (and are not
quantized). Exporting a graph does not prove that TensorRT selects INT8 kernels: check the engine's
layer report. Training and the fake-quant model are unchanged; only a copy is exported.
"""
import copy
import inspect

import torch
from torch import nn
from torch.nn import functional as F

from .cropr_adapter import Cropr
from .integer import folded_bias
from .modules import QuantConv2d, QuantCrossAttention, QuantLinear
from .quantizers import LSQWeightQuantizer


class QDQ(nn.Module):
    """INT8 fake quantization with zero point 0, exported as QuantizeLinear/DequantizeLinear."""

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("zero", torch.zeros((), dtype=torch.int32))

    def forward(self, x):
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero, -128, 127)


class ActivationQDQ(nn.Module):
    """x -> scale * code, where the real offset is removed and returned as `offset` for the bias.

    x_hat (fake quantization) = scale * code + offset for every code, so a layer with bias' = bias +
    offset * sum(weight) computes the same product as fake quantization.
    """

    def __init__(self, quantizer):
        super().__init__()
        scale = quantizer.step.item()
        if not bool(quantizer.initialized) or not 0 < scale < float("inf"):
            raise ValueError("Export requires initialized activation quantizers with a positive step")
        bits, levels = quantizer.bits, 2 ** quantizer.bits
        if quantizer.asymmetric:
            half = 128 if bits == 8 else 0  # Codes [0, 255] become [-128, 127].
            self.offset = quantizer.offset.item() + half * scale
            self.shift = self.offset
            self.low, self.high = (0.0, (levels - 1) * scale) if bits < 8 else (None, None)
        else:
            self.offset = self.shift = 0.0
            self.low, self.high = ((-levels // 2) * scale, (levels // 2 - 1) * scale) if bits < 8 else (None, None)
        self.qdq = QDQ(scale)

    def forward(self, x):
        z = x - self.shift if self.shift else x
        if self.low is not None:
            z = z.clamp(self.low, self.high)
        return self.qdq(z)


def _layer_parts(weight, bias, weight_quant, input_quant):
    """(ActivationQDQ, integer weight codes [out, in], per-channel step, folded bias) of a quantized layer."""
    if not isinstance(weight_quant, LSQWeightQuantizer):
        raise ValueError("TensorRT export supports LSQ/LSQ+ weight quantizers (StatsQ levels are half-integers)")
    activation = ActivationQDQ(input_quant)
    codes, step = weight_quant.integer_codes(weight)
    return activation, codes, step.float(), folded_bias(bias, step, codes, activation.offset)


class ExportLinear(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.activation, codes, step, bias = _layer_parts(module.weight, module.bias, module.weight_quant,
                                                          module.input_quant)
        self.weight_qdq = QDQ(1.0)
        self.register_buffer("codes", codes.float().t().contiguous())  # [in, out]: no Transpose in the graph.
        self.register_buffer("w_step", step)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return torch.matmul(self.activation(x), self.weight_qdq(self.codes)) * self.w_step + self.bias


class ExportConv2d(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.activation, codes, step, bias = _layer_parts(module.weight, module.bias, module.weight_quant,
                                                          module.input_quant)
        self.weight_qdq = QDQ(1.0)
        self.register_buffer("codes", codes.float().reshape(module.weight.shape).contiguous())
        self.register_buffer("w_step", step.view(1, -1, 1, 1))
        self.register_buffer("bias", bias.view(1, -1, 1, 1))
        self.conv = dict(stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)

    def forward(self, x):
        return F.conv2d(self.activation(x), self.weight_qdq(self.codes), **self.conv) * self.w_step + self.bias


class ExportCrossAttention(QuantCrossAttention):
    """Scorer of one Cropr module: scores[b, n] = sum over queries of x_hat_n . q_hat_m."""

    def forward_scorer(self, x):
        product = torch.matmul(self.scorer_activation(self.attn_norm(x)), self.scorer_qdq(self.scorer_codes))
        return (product * self.scorer_step + self.scorer_bias).sum(-1)

    def forward(self, x):
        raise RuntimeError("An export model is for inference only")


class ExportCropr(Cropr):
    """Upstream Cropr inference without operations the ONNX exporter or TensorRT cannot handle."""

    def prune(self, x, scores):
        # The exporter has no argsort overload with stable=False; the default is already False.
        index = torch.argsort(scores, dim=1, descending=True)
        reordered = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        keep = x.shape[1] - self.pruning_rate
        return reordered[:, :keep], reordered[:, keep:]

    def forward(self, x, inference=False):
        if not inference:
            raise RuntimeError("An export model is for inference only")
        scores = self.cross_attn.forward_scorer(x)
        # Equivalent to scores[:, 0] = inf (keep CLS), which is an in-place update the exporter cannot trace.
        keep_cls = torch.full_like(scores[:, :1], float("inf"))
        kept, pruned = self.prune(x, torch.cat([keep_cls, scores[:, 1:]], dim=1))
        return kept, pruned, None


def convert_for_tensorrt(model):
    """Copy a trained B1 model with every quantized layer replaced by its INT8 Q/DQ export form."""
    model = copy.deepcopy(model).cpu().float().eval().requires_grad_(False)
    converted = []
    for name, module in list(model.named_modules()):
        if isinstance(module, QuantLinear):
            replacement = ExportLinear(module)
        elif isinstance(module, QuantConv2d):
            replacement = ExportConv2d(module)
        elif isinstance(module, QuantCrossAttention):
            activation, codes, step, bias = _layer_parts(module.queries[0], None, module.query_quant, module.input_quant)
            module.scorer_activation, module.scorer_qdq = activation, QDQ(1.0)
            module.register_buffer("scorer_codes", codes.float().t().contiguous())
            module.register_buffer("scorer_step", step)
            module.register_buffer("scorer_bias", bias)
            module.__class__ = ExportCrossAttention
            converted.append(name)
            continue
        elif type(module) is Cropr:
            module.__class__ = ExportCropr
            continue
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, replacement)
        converted.append(name)
    if not converted:
        raise ValueError("The model contains no Linear QAT layers to export")
    return model


def export_onnx(model, example, path):
    """Fixed-shape ONNX export (input "images", output "logits"); returns the checked graph.

    Works for the INT8 Q/DQ model and for an FP32 Cropr model: any upstream Cropr module is replaced by
    its exportable form in a copy.
    """
    import onnx

    model = copy.deepcopy(model).cpu().eval()
    for module in model.modules():
        if type(module) is Cropr:
            module.__class__ = ExportCropr
    # Constant folding is off so the graph is the plain trace. The exporter keeps the Q/DQ of constant
    # weights either way; the audit counts them, because they are what makes a product INT8.
    options = dict(opset_version=17, input_names=["images"], output_names=["logits"], do_constant_folding=False)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False  # Newer PyTorch versions default to the dynamo exporter.
    with torch.no_grad():
        torch.onnx.export(model, example.cpu(), str(path), **options)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return graph


def audit_quantized_products(graph):
    """Q/DQ status of every MatMul/Conv: an INT8 product needs both inputs from DequantizeLinear.

    This checks the exported graph, not TensorRT kernel selection.
    """
    producers = {output: node for node in graph.graph.node for output in node.output}
    rows = []
    for node in graph.graph.node:
        if node.op_type not in ("MatMul", "Conv"):
            continue
        dq = [producers.get(name) is not None and producers[name].op_type == "DequantizeLinear"
              for name in node.input[:2]]
        rows.append(dict(name=node.name, op=node.op_type, quantized=all(dq)))
    return rows


def expected_quantized_products(model):
    """Number of quantized MatMul/Conv products in a converted model."""
    return sum(isinstance(m, (ExportLinear, ExportConv2d, ExportCrossAttention)) for m in model.modules())
