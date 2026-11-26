"""Learnable fake quantizers for the operands of Linear layers.

- LSQ: Esser et al., "Learned Step Size Quantization", ICLR 2020 (Eq. 1-3, Appendix B).
- LSQ+: Bhalgat et al., "LSQ+", CVPRW 2020 (learnable offset Eq. 2-4, init Sec. 3.2).
- StatsQ: Liu et al., "Oscillation-free Quantization for Low-bit ViTs", ICML 2023.

Gradients use the papers' detach-based round_pass/grad_scale pseudocode. The output is
the dequantized value in floating point (fake quantization), not a packed integer kernel.
"""
import math

import torch
from torch import nn

BITS = (2, 3, 4, 8)
EPS = 1e-8
# Weight steps are projected to stay above this fraction of their channel's LSQ+ initialization.
MIN_STEP_FRACTION = 0.1
MSE_SAMPLES = 2**18
MSE_GRID = 100


def round_pass(x):
    """Round in the forward pass, identity gradient (STE) in the backward pass."""
    return (x.round() - x).detach() + x


def grad_scale(x, scale):
    """Identity in the forward pass, gradient multiplied by scale in the backward pass."""
    scaled = x * scale
    return (x - scaled).detach() + scaled


def positive(x):
    """Keep a step size positive without blocking its gradient (as in the OFQ reference code).

    Written as clamp + 0 so the forward value is exactly max(x, EPS); x + (clamp - x) loses EPS to
    rounding and returns 0 for x <= -0.25.
    """
    return x.clamp_min(EPS).detach() + (x - x.detach())


def check_bits(bits):
    if bits not in BITS:
        raise ValueError(f"Supported bit widths: {BITS}")


def lsqplus_weight_step(flat, bits):
    """LSQ+ Sec. 3.2.1 weight init, per row: max(|mu - 3 sigma|, |mu + 3 sigma|) / 2^(b-1)."""
    mean, std = flat.mean(1), flat.std(1)
    return torch.maximum((mean - 3 * std).abs(), (mean + 3 * std).abs()) / 2 ** (bits - 1)


class WeightQuantizer(nn.Module):
    """Base class: per-output-channel quantization of a [out, ...] weight tensor."""

    def __init__(self, bits, channels):
        super().__init__()
        check_bits(bits)
        self.bits = bits
        self.channels = channels
        self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

    def project_step(self, weight):
        """No learnable step by default (StatsQ); LSQ overrides this."""

    def collapsed_channels(self, weight):
        """Number of channels whose step is too small to represent the weights; 0 without a step."""
        return 0


class LSQWeightQuantizer(WeightQuantizer):
    """Signed symmetric LSQ with a learnable step size per output channel.

    init="lsq": s = 2 mean|w| / sqrt(Qp) (LSQ Sec. 2.1).
    init="lsq+": s = max(|mu - 3 sigma|, |mu + 3 sigma|) / 2^(b-1) (LSQ+ Sec. 3.2.1).
    """

    def __init__(self, weight, bits, init="lsq+"):
        super().__init__(bits, weight.shape[0])
        if init not in ("lsq", "lsq+"):
            raise ValueError("LSQ weight init must be lsq or lsq+")
        self.init = init
        with torch.no_grad():
            flat = weight.detach().flatten(1)
            if init == "lsq":
                step = 2 * flat.abs().mean(1) / math.sqrt(self.qmax)
            else:
                step = lsqplus_weight_step(flat, bits)
        # 1-D shape: utils.optim.weight_decay_groups places ndim <= 1 parameters in the no-weight-decay group.
        self.step = nn.Parameter(step.clamp_min(EPS).clone())

    def effective_step(self, weight):
        """Per-channel step actually used by forward, so integer conversion uses the same grid."""
        # LSQ g = 1/sqrt(N_W Q_P); N_W is the number of weights sharing this channel's step.
        return positive(grad_scale(self.step, 1.0 / math.sqrt(weight.flatten(1).shape[1] * self.qmax)))

    def forward(self, weight):
        flat = weight.flatten(1)
        step = self.effective_step(weight).unsqueeze(1)
        return (round_pass((flat / step).clamp(self.qmin, self.qmax)) * step).view_as(weight)

    @torch.no_grad()
    def integer_codes(self, weight):
        """Signed integer codes in [qmin, qmax] and the per-channel step: weight_hat = step * codes."""
        flat = weight.flatten(1)
        step = self.effective_step(weight)
        return (flat / step.unsqueeze(1)).clamp(self.qmin, self.qmax).round().to(torch.int8), step

    def step_floor(self, weight):
        return (MIN_STEP_FRACTION * lsqplus_weight_step(weight.detach().flatten(1), self.bits)).clamp_min(EPS)

    @torch.no_grad()
    def project_step(self, weight):
        """Clamp the step after an optimizer update so a channel never collapses.

        An 8-bit step starts near the AdamW learning rate (about 4e-4 vs 5e-4), so a few noisy updates
        can push it to zero or below. positive() then maps it to EPS and the channel quantizes to ~0.
        Steps above the floor are left untouched.
        """
        self.step.copy_(torch.maximum(self.step, self.step_floor(weight)))

    @torch.no_grad()
    def collapsed_channels(self, weight):
        """Channels well below the floor; zero for checkpoints trained with project_step.

        Half the floor, because a channel sitting exactly on the floor is healthy and the floor is
        recomputed from the weights (CUDA and CPU standard deviations differ in the last bits).
        """
        return int((self.step < 0.5 * self.step_floor(weight)).sum())

    def extra_repr(self):
        return f"bits={self.bits}, channels={self.channels}, init={self.init}"


class StatsQWeightQuantizer(WeightQuantizer):
    """OFQ StatsQ: per-channel scale 2 mean|w| recomputed every step; no learnable parameters.

    Levels are symmetric half-integers {-(n-0.5), ..., n-0.5} * scale / n with n = 2^(b-1),
    i.e. 2^b levels without zero, as in the official OFQ implementation.
    """

    def __init__(self, weight, bits):
        super().__init__(bits, weight.shape[0])

    def forward(self, weight):
        flat = weight.flatten(1)
        scale = (2 * flat.abs().mean(1, keepdim=True)).detach().clamp_min(EPS)
        n = 2 ** (self.bits - 1)
        clipped = (flat / scale).clamp(-1, 1 - 1e-6)
        quantized = scale * ((clipped * n - 0.5).round() + 0.5) / n
        # OFQ passes the gradient to every weight (no clipping mask).
        return (flat + (quantized - flat).detach()).view_as(weight)

    def extra_repr(self):
        return f"bits={self.bits}, channels={self.channels}"


class ActivationQuantizer(nn.Module):
    """Per-tensor activation quantizer with a learnable step (static scale, integer-GEMM friendly).

    asymmetric=True (LSQ+): x_hat = clamp(round((x - beta) / s), 0, 2^b - 1) * s + beta with a
    learnable offset beta, initialized by minimizing ||x_hat - x||^2 on the first training batch.
    This forward keeps the offset explicit. Integer execution (integer.py) folds it into the layer
    bias, as in LSQ+ Eq. 5.
    asymmetric=False (LSQ): signed range [-2^(b-1), 2^(b-1) - 1], s = 2 mean|x| / sqrt(Qp).
    """

    def __init__(self, bits, asymmetric=True):
        super().__init__()
        check_bits(bits)
        self.bits = bits
        self.asymmetric = asymmetric
        if asymmetric:
            self.qmin, self.qmax = 0, 2**bits - 1
            self.offset = nn.Parameter(torch.zeros(1))
        else:
            self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            self.register_parameter("offset", None)
        self.step = nn.Parameter(torch.ones(1))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("initial_step", torch.ones(1))  # Reference for project_step, fixed at initialization.
        self._ready = False  # Mirrors the buffer to avoid a device sync every forward.

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before initial_step existed use the loaded step as their own reference.
        if prefix + "initial_step" not in state_dict and prefix + "step" in state_dict:
            state_dict[prefix + "initial_step"] = state_dict[prefix + "step"].detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ready = False

    @torch.no_grad()
    def initialize(self, x):
        x = x.detach().float().flatten()
        if not self.asymmetric:
            self.step.fill_(max(2 * x.abs().mean().item() / math.sqrt(self.qmax), EPS))
        else:
            step, offset = mse_asymmetric_range(x, self.qmax)
            self.step.fill_(step)
            self.offset.fill_(offset)
        self.initial_step.copy_(self.step)
        self.initialized.fill_(True)

    def forward(self, x):
        if not self._ready:
            if not bool(self.initialized):
                if not self.training:
                    raise RuntimeError("Activation quantizer needs a training batch before evaluation")
                self.initialize(x)
            self._ready = True
        # LSQ g = 1/sqrt(N_F Q_P) with N_F = x.numel() as in the Q-ViT, OFQ and PyTorch
        # (_LearnableFakeQuantize) implementations.
        g = 1.0 / math.sqrt(x.numel() * self.qmax)
        step = positive(grad_scale(self.step, g))
        if self.offset is None:
            return round_pass((x / step).clamp(self.qmin, self.qmax)) * step
        # The offset sum spans the same elements as the step gradient, so it uses the same g.
        offset = grad_scale(self.offset, g)
        return round_pass(((x - offset) / step).clamp(self.qmin, self.qmax)) * step + offset

    @torch.no_grad()
    def project_step(self):
        """Clamp the step after an optimizer update (see LSQWeightQuantizer.project_step).

        A layer input such as attn.proj can start at an 8-bit step near 4e-4, about the AdamW learning
        rate, so a few updates can collapse the whole tensor. Activations have no per-batch statistic
        that stays fixed at inference, so the floor is relative to the initial step.
        """
        self.step.copy_(torch.maximum(self.step, (MIN_STEP_FRACTION * self.initial_step).clamp_min(EPS)))

    @torch.no_grad()
    def collapsed(self):
        """True when the step is far below the floor (a channel on the floor is healthy)."""
        return self.step.item() <= max(1e-6, 0.5 * MIN_STEP_FRACTION * self.initial_step.item())

    def extra_repr(self):
        return f"bits={self.bits}, asymmetric={self.asymmetric}"


@torch.no_grad()
def mse_asymmetric_range(x, qmax, samples=MSE_SAMPLES, grid=MSE_GRID):
    """LSQ+ Eq. 8 restricted to shrunk min/max ranges; ratio 1 is the LSQ+ Eq. 7 min/max init."""
    low, high = x.min(), x.max()
    if x.numel() > samples:  # Deterministic strided subset; min/max still use the full tensor.
        x = x[torch.linspace(0, x.numel() - 1, samples, device=x.device).long()]
    best = None
    for ratio in torch.linspace(1.0 / grid, 1.0, grid).tolist():
        a, b = low * ratio, high * ratio
        step = ((b - a) / qmax).clamp_min(EPS)
        error = (((x - a) / step).round().clamp(0, qmax) * step + a - x).square().mean()
        if best is None or error < best[0]:
            best = (error, step.item(), a.item())
    return best[1], best[2]


def make_weight_quantizer(weight, bits, method):
    if method == "statsq":
        return StatsQWeightQuantizer(weight, bits)
    return LSQWeightQuantizer(weight, bits, init=method)


def make_activation_quantizer(bits, method):
    if method not in ("lsq", "lsq+"):
        raise ValueError("Activation quantizer must be lsq or lsq+")
    return ActivationQuantizer(bits, asymmetric=method == "lsq+")
