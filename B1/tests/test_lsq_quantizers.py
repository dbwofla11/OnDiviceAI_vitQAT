"""Quantizer checks against the LSQ/LSQ+/StatsQ formulas and PyTorch's learnable fake-quant kernels."""
import math

import pytest
import torch

from B1.quantizers import (EPS, MIN_STEP_FRACTION, ActivationQuantizer, LSQWeightQuantizer, StatsQWeightQuantizer,
                           mse_asymmetric_range, positive)


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def away_from_clip_band(ratio, qmin, qmax):
    """Move values out of (qmax, qmax + 0.5) and (qmin - 0.5, qmin).

    LSQ (Eq. 1) clips before rounding, while PyTorch's kernel masks on the rounded value, so the
    two definitions differ only inside these half-step bands outside the clipping range.
    """
    ratio = torch.where((ratio > qmax) & (ratio < qmax + 0.5), ratio + 0.5, ratio)
    return torch.where((ratio < qmin) & (ratio > qmin - 0.5), ratio - 0.5, ratio)


def lsq_reference(v, s, qmin, qmax):
    """LSQ Eq. 1-3 and Eq. 5 written out explicitly: (v_hat, dv_hat/dv, dv_hat/ds).

    Autograd computes dv_hat/dv as (upstream * s) / s, so comparisons allow float rounding.
    """
    r = v / s
    inside = (r > qmin) & (r < qmax)
    v_hat = r.clamp(qmin, qmax).round() * s
    ds = torch.where(inside, r.round() - r, torch.where(r <= qmin, float(qmin), float(qmax)))
    return v_hat, inside.to(v.dtype), ds


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("init", ["lsq", "lsq+"])
def test_weight_lsq_matches_paper_and_pytorch_kernel(bits, init):
    torch.manual_seed(bits)
    weight = torch.randn(6, 40) * torch.linspace(0.5, 2, 6).unsqueeze(1)
    quantizer = LSQWeightQuantizer(weight, bits, init=init)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    flat = weight
    if init == "lsq":
        expected = 2 * flat.abs().mean(1) / math.sqrt(qmax)
    else:
        mean, std = flat.mean(1), flat.std(1)
        expected = torch.maximum((mean - 3 * std).abs(), (mean + 3 * std).abs()) / 2 ** (bits - 1)
    assert torch.allclose(quantizer.step, expected)
    step = quantizer.step.detach().unsqueeze(1)
    ratio = away_from_clip_band(torch.empty_like(weight).uniform_(qmin - 3, qmax + 3), qmin, qmax)
    w = (ratio * step).requires_grad_()
    upstream = torch.randn_like(w)
    out = quantizer(w)
    (out * upstream).sum().backward()

    v_hat, dv, ds = lsq_reference(w.detach(), step, qmin, qmax)
    g = 1 / math.sqrt(w.shape[1] * qmax)
    assert torch.allclose(out, v_hat)
    assert torch.allclose(w.grad, upstream * dv) and torch.equal(w.grad == 0, dv == 0)
    assert torch.allclose(quantizer.step.grad, (upstream * ds).sum(1) * g, rtol=1e-5, atol=1e-6)
    assert all(row.unique().numel() <= 2**bits for row in out)

    # Same values and gradients as torch.ao's LSQ kernel (_LearnableFakeQuantize).
    scale = quantizer.step.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    reference = torch._fake_quantize_learnable_per_channel_affine(
        w_ref, scale, torch.zeros(6), 0, qmin, qmax, g)
    (reference * upstream).sum().backward()
    assert torch.allclose(out, reference)
    assert torch.allclose(w.grad, w_ref.grad)
    assert torch.allclose(quantizer.step.grad, scale.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_symmetric_activation_lsq_matches_paper_and_pytorch_kernel(bits):
    torch.manual_seed(bits)
    quantizer = ActivationQuantizer(bits, asymmetric=False)
    first = torch.randn(4, 9, 16)
    quantizer(first)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    assert quantizer.step.item() == pytest.approx(2 * first.abs().mean().item() / math.sqrt(qmax))
    s = quantizer.step.detach()
    x = (away_from_clip_band(torch.empty(4, 9, 16).uniform_(qmin - 3, qmax + 3), qmin, qmax) * s).requires_grad_()
    upstream = torch.randn_like(x)
    out = quantizer(x)
    (out * upstream).sum().backward()
    v_hat, dv, ds = lsq_reference(x.detach(), s, qmin, qmax)
    g = 1 / math.sqrt(x.numel() * qmax)
    assert torch.allclose(out, v_hat)
    assert torch.allclose(x.grad, upstream * dv) and torch.equal(x.grad == 0, dv == 0)
    assert quantizer.step.grad.item() == pytest.approx(((upstream * ds).sum() * g).item(), rel=1e-5)

    scale = s.clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    reference = torch._fake_quantize_learnable_per_tensor_affine(x_ref, scale, torch.zeros(1), qmin, qmax, g)
    (reference * upstream).sum().backward()
    assert torch.allclose(out, reference)
    assert torch.allclose(x.grad, x_ref.grad)
    assert quantizer.step.grad.item() == pytest.approx(scale.grad.item(), rel=1e-5)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_asymmetric_activation_matches_lsq_plus(bits):
    torch.manual_seed(10 + bits)
    quantizer = ActivationQuantizer(bits, asymmetric=True)
    gelu_like = torch.nn.functional.gelu(torch.randn(8, 5, 32) * 2)
    quantizer(gelu_like)  # first batch initializes step and offset
    s, beta = quantizer.step.detach(), quantizer.offset.detach()
    qmax = 2**bits - 1
    x = (torch.empty(8, 5, 32).uniform_(-3, qmax + 3) * s + beta).requires_grad_()
    upstream = torch.randn_like(x)
    out = quantizer(x)
    (out * upstream).sum().backward()
    # LSQ+ Eq. 2-4 with the LSQ gradient scale.
    r = (x.detach() - beta) / s
    inside = (r > 0) & (r < qmax)
    g = 1 / math.sqrt(x.numel() * qmax)
    ds = torch.where(inside, r.round() - r, torch.where(r <= 0, 0.0, float(qmax)))
    dbeta = (~inside).float()
    levels = (out.detach() - beta) / s
    assert torch.allclose(levels, levels.round(), atol=1e-4)
    assert levels.min() >= -1e-4 and levels.max() <= qmax + 1e-4
    assert torch.allclose(x.grad, upstream * inside) and torch.equal(x.grad != 0, inside)
    assert quantizer.step.grad.item() == pytest.approx(((upstream * ds).sum() * g).item(), rel=1e-4, abs=1e-6)
    assert quantizer.offset.grad.item() == pytest.approx(((upstream * dbeta).sum() * g).item(), rel=1e-4, abs=1e-6)


def test_lsq_plus_mse_initialization():
    torch.manual_seed(0)
    x = torch.randn(4096)
    # A single candidate is the LSQ+ Eq. 7 min/max initialization.
    step, offset = mse_asymmetric_range(x, 3, grid=1)
    assert step == pytest.approx(((x.max() - x.min()) / 3).item())
    assert offset == pytest.approx(x.min().item())

    def error(step, offset):
        return (((x - offset) / step).round().clamp(0, 3) * step + offset - x).square().mean().item()

    mse_step, mse_offset = mse_asymmetric_range(x, 3)
    assert error(mse_step, mse_offset) < 0.5 * error(step, offset)  # 2-bit: clipping beats min/max


def test_activation_initialization_once_state_and_eval_guard():
    quantizer = ActivationQuantizer(2)
    with pytest.raises(RuntimeError, match="training batch"):
        quantizer.eval()(torch.randn(10))
    quantizer.train()(torch.randn(100))
    state = {k: v.clone() for k, v in quantizer.state_dict().items()}
    assert state["initialized"].item()
    quantizer(torch.randn(100) * 50)  # later batches never re-initialize
    assert torch.equal(quantizer.step, state["step"]) and torch.equal(quantizer.offset, state["offset"])
    restored = ActivationQuantizer(2)
    restored.load_state_dict(quantizer.state_dict())
    x = torch.randn(32)
    assert torch.equal(restored.eval()(x), quantizer.eval()(x))
    stale = ActivationQuantizer(2)
    stale.train()(torch.randn(100))
    stale.load_state_dict(ActivationQuantizer(2).state_dict())  # loading resets the cached flag
    with pytest.raises(RuntimeError, match="training batch"):
        stale.eval()(x)


def ofq_statsq_reference(real_weights, num_bits):
    """Copied from the official OFQ StatsQuantizer.forward (clip_val = 2)."""
    scaling_factor = 2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
    scaled_weights = real_weights / scaling_factor
    cliped_weights = torch.clamp(scaled_weights, min=-1.0, max=1.0 - 1e-6)
    n = float(2 ** (num_bits - 1))
    quan_weights_no_grad = scaling_factor * ((torch.round(cliped_weights * n - 0.5) + 0.5) / n)
    return quan_weights_no_grad.detach() - real_weights.detach() + real_weights


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_statsq_matches_ofq_reference(bits):
    torch.manual_seed(bits)
    weight = (torch.randn(8, 50) * 0.02).requires_grad_()
    quantizer = StatsQWeightQuantizer(weight, bits)
    out = quantizer(weight)
    assert list(quantizer.parameters()) == []
    assert torch.allclose(out, ofq_statsq_reference(weight, bits))
    scale = 2 * weight.detach().abs().mean(1, keepdim=True)
    levels = out.detach() / scale * 2 ** (bits - 1)  # half-integers in [-(n - 0.5), n - 0.5]
    assert torch.allclose(levels, (levels - 0.5).round() + 0.5, atol=1e-4)
    assert levels.abs().max() <= 2 ** (bits - 1) - 0.5 + 1e-4
    upstream = torch.randn_like(weight)
    (out * upstream).sum().backward()
    assert torch.equal(weight.grad, upstream)


def test_quantizer_parameters_are_one_dimensional():
    # utils.optim.weight_decay_groups excludes ndim <= 1 parameters from weight decay.
    modules = [LSQWeightQuantizer(torch.randn(3, 4), 2), ActivationQuantizer(2), ActivationQuantizer(2, False)]
    assert all(p.ndim == 1 for module in modules for p in module.parameters())
    with pytest.raises(ValueError, match="bit"):
        ActivationQuantizer(5)


@pytest.mark.parametrize("value", [1.0, 0.02, 1e-9, 0.0, -1e-9, -1e-4, -0.02, -0.2, -0.25, -0.3, -0.5, -1.0, -100.0])
def test_positive_is_exactly_the_clamped_value_with_identity_gradient(value):
    # x + (clamp - x) rounds to 0 for x <= -0.25, which then divides by zero in the quantizers.
    x = torch.tensor([value], requires_grad=True)
    out = positive(x)
    assert out.item() == max(x.item(), torch.tensor(EPS).item()) and out.item() > 0
    out.sum().backward()
    assert x.grad.item() == 1.0


@pytest.mark.parametrize("init", ["lsq", "lsq+"])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_step_projection_restores_only_collapsed_channels(bits, init):
    torch.manual_seed(0)
    weight = torch.randn(6, 400) * torch.linspace(0.005, 0.05, 6).unsqueeze(1)
    quantizer = LSQWeightQuantizer(weight, bits, init=init)
    healthy = quantizer.step.detach().clone()
    quantizer.project_step(weight)  # A fresh quantizer is never below the floor.
    assert torch.equal(quantizer.step, healthy) and quantizer.collapsed_channels(weight) == 0
    with torch.no_grad():
        quantizer.step[0], quantizer.step[3] = -0.5, 1e-12  # As after a too large optimizer update.
    assert quantizer.collapsed_channels(weight) == 2
    mean, std = weight.mean(1), weight.std(1)
    floor = MIN_STEP_FRACTION * torch.maximum((mean - 3 * std).abs(), (mean + 3 * std).abs()) / 2 ** (bits - 1)
    quantizer.project_step(weight)
    assert quantizer.collapsed_channels(weight) == 0 and (quantizer.step > 0).all()
    for channel in range(6):
        assert torch.equal(quantizer.step[channel], floor[channel] if channel in (0, 3) else healthy[channel])
    out = quantizer(weight)
    assert torch.isfinite(out).all() and (out[0].abs().max() > 0).item()


def test_statsq_has_no_step_to_project_and_activation_step_stays_positive():
    statsq = StatsQWeightQuantizer(torch.randn(4, 10), 2)
    statsq.project_step(torch.randn(4, 10))
    assert statsq.collapsed_channels(torch.randn(4, 10)) == 0
    activation = ActivationQuantizer(8)
    activation.train()(torch.randn(100) * 1e-2)  # A small-scale layer input: the step starts near the lr.
    assert torch.equal(activation.initial_step, activation.step)
    before = activation.step.detach().clone()
    activation.project_step()
    assert torch.equal(activation.step, before) and not activation.collapsed()
    for broken in (-0.2, 0.0, 1e-9):
        activation.step.data.fill_(broken)
        assert activation.collapsed()
        activation.project_step()  # Floor relative to the initial step, which stays fixed.
        assert activation.step.item() == pytest.approx(MIN_STEP_FRACTION * activation.initial_step.item(), rel=1e-6)
        assert not activation.collapsed()
        assert torch.isfinite(activation.eval()(torch.randn(100))).all()


def test_activation_initial_step_is_saved_and_older_checkpoints_restore():
    quantizer = ActivationQuantizer(4)
    quantizer.train()(torch.randn(50) * 3)
    state = quantizer.state_dict()
    assert torch.equal(state["initial_step"], state["step"])
    quantizer.step.data.mul_(0.5)  # Later training moves the step, not its reference.
    assert quantizer.initial_step.item() != quantizer.step.item()
    old_state = {key: value.clone() for key, value in quantizer.state_dict().items() if key != "initial_step"}
    restored = ActivationQuantizer(4)
    restored.load_state_dict(old_state)  # strict
    assert restored.initial_step.item() == restored.step.item() == quantizer.step.item()
    assert not restored.collapsed()


def test_effective_step_and_integer_codes_match_the_forward_grid():
    torch.manual_seed(1)
    weight = torch.randn(5, 64) * 0.02
    for bits in (2, 4, 8):
        quantizer = LSQWeightQuantizer(weight, bits)
        codes, step = quantizer.integer_codes(weight)
        assert codes.dtype == torch.int8 and codes.min() >= quantizer.qmin and codes.max() <= quantizer.qmax
        assert torch.equal(quantizer(weight).detach(), codes.float() * step.unsqueeze(1))
