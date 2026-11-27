"""B1 (LSQ/LSQ+) model as an INT8 Q/DQ graph for TensorRT, built from the shared utils.int8 blocks.

The learned grids are kept: utils.int8 explains how offsets and 8-bit codes are carried with zero-point-0
INT8. This module only reads the LSQ quantizers and handles the B1-specific parts: the Cropr scorer and the
upstream Cropr operations the ONNX exporter cannot trace. Training and the fake-quant model are unchanged;
only a copy is exported.
"""
import copy

import torch

from utils.int8 import Int8Conv2d, Int8Linear, WeightCodes, lsq_grid
from .cropr_adapter import Cropr
from .modules import QuantConv2d, QuantCrossAttention, QuantLinear
from .quantizers import LSQWeightQuantizer


def activation_grid(quantizer):
    """Grid of a calibrated B1 ActivationQuantizer."""
    step = quantizer.step.item()
    if not bool(quantizer.initialized) or not 0 < step < float("inf"):
        raise ValueError("Export requires initialized activation quantizers with a positive step")
    offset = quantizer.offset.item() if quantizer.asymmetric else 0.0
    return lsq_grid(step, offset, quantizer.bits, quantizer.asymmetric)


def weight_codes(weight, quantizer):
    if not isinstance(quantizer, LSQWeightQuantizer):
        raise ValueError("TensorRT export supports LSQ/LSQ+ weight quantizers (StatsQ levels are half-integers)")
    codes, step = quantizer.integer_codes(weight)
    return WeightCodes(codes, step.float())


class ExportCrossAttention(QuantCrossAttention):
    """Scorer of one Cropr module: scores[b, n] = sum over queries of x_hat_n . q_hat_m."""

    def forward_scorer(self, x):
        return self.scorer(self.attn_norm(x)).sum(-1)

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
        # Equivalent to scores[:, 0] = inf (keep CLS), an in-place update the exporter cannot trace.
        keep_cls = torch.full_like(scores[:, :1], float("inf"))
        kept, pruned = self.prune(x, torch.cat([keep_cls, scores[:, 1:]], dim=1))
        return kept, pruned, None


def exportable(model):
    """Copy whose upstream Cropr modules use the exportable inference (also for the FP32 reference)."""
    model = copy.deepcopy(model).cpu().float().eval().requires_grad_(False)
    for module in model.modules():
        if type(module) is Cropr:
            module.__class__ = ExportCropr
    return model


def convert_for_tensorrt(model):
    """Copy of a trained B1 model with every quantized layer replaced by its INT8 Q/DQ form."""
    model = exportable(model)
    converted = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, QuantLinear):
            replacement = Int8Linear(activation_grid(module.input_quant),
                                     weight_codes(module.weight, module.weight_quant), module.bias)
        elif isinstance(module, QuantConv2d):
            replacement = Int8Conv2d(module, activation_grid(module.input_quant),
                                     weight_codes(module.weight, module.weight_quant))
        elif isinstance(module, QuantCrossAttention):
            module.scorer = Int8Linear(activation_grid(module.input_quant),
                                       weight_codes(module.queries[0], module.query_quant))
            del module.input_quant, module.query_quant  # Replaced by the INT8 scorer.
            module.__class__ = ExportCrossAttention
            converted += 1
            continue
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, replacement)
        converted += 1
    if not converted:
        raise ValueError("The model contains no Linear QAT layers to export")
    return model
