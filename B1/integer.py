"""Run a trained B1 model with integer (INT8) kernels on CPU, to measure integer latency.

Accuracy is evaluated with the fake-quant model. This module copies that model and replaces every
quantized Linear, patch embedding and Cropr scorer by a u8 x s8 integer matrix product (int32
accumulation, FP32 output). The learned LSQ/LSQ+ grids are unchanged:

    y = s_x * s_w * (x_u8 @ w_s8^T) + bias'        x_u8 = clamp(round((x - beta) / s_x), 0, 2^b - 1)
    bias' = bias + beta * s_w * sum(w_s8)          (LSQ+ Eq. 5; symmetric LSQ shifts codes by 2^(b-1))

The real-valued offset therefore needs no integer zero point and no remapping of the activation grid.
For the same layer input, every integer code equals the fake-quant code and the layer output agrees to
float32 rounding (about 1e-6). Logits of the whole model are close but not bit-identical: rounding
differences between layers can move a value across a code boundary.

Weights of any bit width are stored in INT8, so the latency is that of an INT8 kernel: it does not
show the speed or memory benefit of packed 2/4-bit weights. Attention matmuls, softmax, LayerNorm,
GELU and residual adds stay FP32, as in fake-quant evaluation.

PyTorch has no CUDA INT8 kernel, so this path is CPU only.
"""
import copy
import functools

import torch
from torch import nn

from .modules import QuantConv2d, QuantCrossAttention, QuantLinear, weight_quantizers
from .quantizers import LSQWeightQuantizer

BACKEND = "onednn_u8s8_int32"


def _kernel(u8, scale, packed, w_step, w_zero, bias):
    """int8 GEMM with FP32 output: scale * w_step * (u8 @ s8) + bias."""
    return torch.ops.onednn.qlinear_pointwise(u8, scale, 0, packed, w_step, w_zero, bias, 1.0, 0,
                                              torch.float32, "none", [], "")


# (rows, in features, out features) of the layers B1 uses: block Linear, head/scorer-like outputs.
_CHECK_SHAPES = ((64, 768, 192), (197, 192, 768), (1, 192, 101), (197, 192, 1))


@functools.lru_cache(maxsize=None)
def integer_kernel_error():
    """None when the kernel is available and exact on this CPU, otherwise the reason.

    Without VNNI, u8 x s8 products can saturate int16. The default x86/fbgemm kernel then returns wrong
    8-bit results without any error, so the integer result is compared with exact integer arithmetic.
    """
    try:
        generator = torch.Generator().manual_seed(0)
        for rows, features, outputs in _CHECK_SHAPES:
            for low, high in ((-128, 127), (-2, 1)):
                peak = 255 if low == -128 else 3
                x = torch.randint(0, peak + 1, (rows, features), generator=generator)
                inputs = (x, torch.full_like(x, peak))
                weights = (torch.full((outputs, features), high), torch.full((outputs, features), low),
                           torch.randint(low, high + 1, (outputs, features), generator=generator))
                step = torch.linspace(0.001, 0.01, outputs)
                zero = torch.zeros(outputs, dtype=torch.long)
                bias = torch.linspace(-1, 1, outputs)
                for codes in weights:
                    packed = torch.ops.onednn.qlinear_prepack(codes.to(torch.int8), None)
                    for x_codes in inputs:
                        y = _kernel(x_codes.to(torch.uint8), 0.03, packed, step, zero, bias)
                        exact = (x_codes.double() @ codes.double().t()) * (0.03 * step.double()) + bias.double()
                        if not torch.isfinite(y).all() or (y - exact).abs().max() > 1e-5 * exact.abs().max():
                            return "this CPU's integer kernel does not reproduce exact integer arithmetic"
    except Exception as error:  # Missing op/build: any failure means the kernel cannot be used here.
        return f"integer kernel unavailable: {str(error).splitlines()[0]}"
    return None


def folded_bias(bias, step, codes, offset):
    """bias + offset * sum_j w_ij: folds a constant activation offset into the bias (LSQ+ Eq. 5).

    codes [out, in] are the signed integer weight codes (weight_hat = step * codes), so the sum is exact.
    """
    folded = offset * step.double() * codes.double().sum(1)
    bias = torch.zeros(len(step), dtype=torch.float64) if bias is None else bias.detach().double()
    return (bias + folded).float()


def unsupported_reason(model):
    """None if this model can run with integer kernels on this machine, otherwise the reason."""
    error = integer_kernel_error()
    if error:
        return error
    for _, quantizer in weight_quantizers(model):
        if not isinstance(quantizer, LSQWeightQuantizer):
            return ("integer execution supports LSQ/LSQ+ weight quantizers; "
                    "StatsQ levels are half-integers")
    return None


class IntegerMatmul(nn.Module):
    """FP32 input -> u8 codes -> int8 GEMM -> FP32 output, for one quantized weight matrix [out, in]."""

    def __init__(self, weight, bias, weight_quant, input_quant):
        super().__init__()
        if not isinstance(weight_quant, LSQWeightQuantizer):
            raise ValueError("Integer execution supports LSQ/LSQ+ weight quantizers")
        if not bool(input_quant.initialized):
            raise ValueError("Integer execution requires an initialized activation quantizer")
        scale = input_quant.step.item()
        if not 0 < scale < float("inf"):
            raise ValueError("Integer execution requires a positive activation step")
        codes, step = weight_quant.integer_codes(weight)
        # Input codes are u8: asymmetric x_hat = s u + beta; symmetric x_hat = s (u - z), z = 2^(b-1).
        self.scale = scale
        self.qmin, self.qmax = input_quant.qmin, input_quant.qmax
        if input_quant.asymmetric:
            self.shift, self.zero, offset = input_quant.offset.item(), 0, input_quant.offset.item()
        else:
            self.zero = 2 ** (input_quant.bits - 1)
            self.shift, offset = 0.0, -scale * self.zero
        self.register_buffer("w_step", step.float())
        self.register_buffer("w_zero", torch.zeros(len(step), dtype=torch.long))
        self.register_buffer("bias", folded_bias(bias, step, codes, offset))
        self.packed = torch.ops.onednn.qlinear_prepack(codes, None)

    def quantize(self, x):
        """u8 codes computed with the same division and rounding as ActivationQuantizer.forward."""
        z = (x - self.shift).div_(self.scale) if self.shift else x / self.scale
        z = z.clamp_(self.qmin, self.qmax).round_()
        return (z.add_(self.zero) if self.zero else z).to(torch.uint8)

    def matmul(self, u8):
        return _kernel(u8, self.scale, self.packed, self.w_step, self.w_zero, self.bias)

    def forward(self, x):
        return self.matmul(self.quantize(x))


class IntegerLinear(IntegerMatmul):
    def __init__(self, module):
        super().__init__(module.weight, module.bias, module.weight_quant, module.input_quant)


class IntegerPatchEmbed(IntegerMatmul):
    """A stride-equal-to-kernel convolution, computed as a matrix product over the image patches."""

    def __init__(self, module):
        if (module.kernel_size != module.stride or module.padding != (0, 0) or module.dilation != (1, 1)
                or module.groups != 1):
            raise ValueError("Integer patch embedding requires kernel == stride, no padding or dilation, groups 1")
        super().__init__(module.weight, module.bias, module.weight_quant, module.input_quant)
        self.out_channels, self.patch = module.out_channels, module.kernel_size[0]

    def forward(self, x):
        u8, p = self.quantize(x), self.patch
        b, c, h, w = u8.shape
        patches = u8.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * p * p)
        return self.matmul(patches).transpose(1, 2).reshape(b, self.out_channels, h // p, w // p)


class IntegerCrossAttention(QuantCrossAttention):
    """Cropr scorer with integer dot products; the training forward is not supported."""

    def forward_scorer(self, x):
        return self.integer_scorer(self.attn_norm(x)).sum(-1)

    def forward(self, x):
        raise RuntimeError("An integer model is for inference only")


def to_integer(model):
    """Copy a trained B1 model with all quantized layers replaced by integer kernels (CPU, eval mode)."""
    reason = unsupported_reason(model)
    if reason:
        raise RuntimeError(reason)
    model = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    for name, module in list(model.named_modules()):
        if isinstance(module, QuantLinear):
            replacement = IntegerLinear(module)
        elif isinstance(module, QuantConv2d):
            replacement = IntegerPatchEmbed(module)
        elif isinstance(module, QuantCrossAttention):
            module.integer_scorer = IntegerMatmul(module.queries[0], None, module.query_quant, module.input_quant)
            module.__class__ = IntegerCrossAttention
            continue
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, replacement)
    return model
