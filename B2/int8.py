"""B2 (affine QAT with a quantized Cropr scorer) as an INT8 Q/DQ graph, built from the shared utils.int8 blocks.

utils.int8.convert_affine_qat converts every Linear/Conv and the backbone attention. This module adds the
B2-specific parts: the Cropr scorer, whose quantized queries are constant at inference, and the upstream
Cropr operations the ONNX exporter cannot trace. Training and the fake-quant model are unchanged.
"""
import copy

import torch

from utils.int8 import Int8Linear, affine_grid, convert_affine_qat, weight_codes
from .cropr_adapter import Cropr
from .quantization import QATCrossAttention


class ExportCrossAttention(QATCrossAttention):
    """Scorer of one Cropr module: scores[b, n] = sum over queries of k_hat_n . q_hat_m."""

    def forward_scorer(self, x):
        return self.scorer(self.k(self.attn_norm(x))).sum(-1)

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


def convert_for_tensorrt(model):
    """Copy of a trained B2 model with every quantized product in INT8 form (utils.int8)."""
    model = copy.deepcopy(model).cpu().float().eval().requires_grad_(False)
    for module in list(model.modules()):
        if isinstance(module, QATCrossAttention):
            with torch.no_grad():  # q_hat = q_quant(q(query_quant(queries))) does not depend on the image.
                queries = module.q_quant(module.q(module.query_quant(module.queries)))[0]
            q_fake_quant = module.q_quant.fake_quant
            module.scorer = Int8Linear(affine_grid(module.k_quant),
                                       weight_codes(queries, q_fake_quant.scale, q_fake_quant.zero_point))
            for name in ("query_quant", "q_quant", "k_quant", "v_quant", "prob_quant"):
                delattr(module, name)  # Replaced by the INT8 scorer; no fake quantizer stays in the export.
            module.__class__ = ExportCrossAttention
        elif type(module) is Cropr:
            module.__class__ = ExportCropr
    return convert_affine_qat(model)
