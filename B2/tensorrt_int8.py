"""Export-only INT8 Q/DQ: preserve W2 values, adapt A8 to symmetric grids.

The PyTorch model is a numerical reference, not an accelerated runtime. TensorRT
selects the execution kernels from the exported ONNX graph.
"""
import copy
import inspect

import torch
from torch import nn

from .cropr_adapter import Cropr
from .quantization import QATAttention, Quantizer


class FrozenQDQ(nn.Module):
    def __init__(self, scale, per_channel=False):
        super().__init__()
        self.per_channel = per_channel
        self.register_buffer("scale", scale.detach().float().clone())
        self.register_buffer("zero", torch.zeros_like(scale, dtype=torch.int32))

    def forward(self, x):
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(x, self.scale, self.zero, 0, -128, 127)
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero, -128, 127)


class ExportCropr(Cropr):
    def prune(self, x, scores):
        # Omitting stable=False selects the argsort overload supported by the
        # legacy ONNX exporter. Its value is already False by default.
        idx = torch.argsort(scores, dim=1, descending=True)
        reordered = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        keep = x.shape[1] - self.pruning_rate
        return reordered[:, :keep], reordered[:, keep:]


class INT8Attention(QATAttention):
    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Put scaling before Q/DQ, so both BMM operands come from INT8 DQ.
        # q_quant.scale is multiplied by the same factor during conversion.
        q = self.q_quant(self.q_norm(q) * self.scale)
        k, v = self.k_quant(self.k_norm(k)), self.v_quant(v)
        prob = self.prob_quant(self.attn_drop((q @ k.transpose(-2, -1)).softmax(dim=-1)))
        x = (prob @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


@torch.no_grad()
def convert_tensorrt_int8(source):
    """Copy calibrated W2A8 QAT; freeze W2 and remap every A8 quantizer.

    A8 scales cover the old representable range using symmetric signed INT8.
    This changes activation rounding/clipping, including attention probabilities.
    It is NOT numerically equivalent to the original affine W2A8 model.
    """
    model = copy.deepcopy(source).cpu().float().eval()
    sites = [(name, module) for name, module in model.named_modules() if isinstance(module, Quantizer)]
    if not sites:
        raise ValueError("A calibrated W2A8 QAT model is required")
    for name, quant in sites:
        parent_name, _, role = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        weight = role in ("weight_quant", "query_quant")
        fq = quant.fake_quant
        observed = fq.activation_post_process.min_val
        if (quant.bits != (2 if weight else 8) or not bool(fq.fake_quant_enabled.item())
                or observed.numel() == 0 or not torch.isfinite(observed).all()
                or not torch.isfinite(fq.scale).all() or not (fq.scale > 0).all()):
            raise ValueError(f"{name}: enabled, calibrated W2/A8 quantization is required")
        if not ((fq.zero_point >= 0) & (fq.zero_point <= (3 if weight else 255))).all():
            raise ValueError(f"{name}: zero point outside the original quantization range")
        if weight:
            parameter = parent.weight if role == "weight_quant" else parent.queries
            # Bake the original affine W2 grid before replacing its zero point.
            parameter.copy_(quant(parameter))
            scale = fq.scale
        else:
            # Preserve range coverage, not the original asymmetric quantization grid.
            scale = fq.scale * torch.maximum(fq.zero_point, 255 - fq.zero_point).float() / 127
        setattr(parent, role, FrozenQDQ(scale, per_channel=weight))
    for module in model.modules():
        if isinstance(module, QATAttention):
            module.__class__ = INT8Attention
            module.q_quant.scale.mul_(module.scale)
    model.requires_grad_(False)
    return model


def export_onnx(model, example, path):
    """Fixed shape export compatible with the project's PyTorch 2.3 environment."""
    import onnx

    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if isinstance(module, Cropr):
            module.__class__ = ExportCropr
    options = dict(opset_version=17, input_names=["images"], output_names=["logits"],
                   do_constant_folding=True)
    # New PyTorch versions default to dynamo; the 2.3 exporter has no such option.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(model.eval(), example, str(path), **options)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return graph


def audit_attention_qdq(graph):
    """Check exported attention BMM operands, not TensorRT kernel selection."""
    producers = {out: node for node in graph.graph.node for out in node.output}

    def from_dq(name):
        node = producers.get(name)
        while node is not None and node.op_type in ("Transpose", "Reshape", "Identity", "Expand"):
            node = producers.get(node.input[0])
        return node is not None and node.op_type == "DequantizeLinear"

    # Projection nodes live in /qkv/ or /proj/; attention BMMs are immediate children.
    rows = []
    for node in graph.graph.node:
        parent = node.name.rsplit("/", 1)[0]
        if node.op_type == "MatMul" and (parent.endswith(("/attn", "/cross_attn"))
                                        or parent.rsplit("/", 1)[-1].startswith("cropr.")):
            rows.append(dict(name=node.name, both_inputs_qdq=all(from_dq(x) for x in node.input)))
    return rows
