"""Integer (INT8 kernel) execution: exact codes, close outputs, guards and latency metrics."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from B1 import integer
from B1.integer import (BACKEND, IntegerCrossAttention, IntegerLinear, IntegerMatmul, IntegerPatchEmbed,
                        integer_kernel_error, to_integer, unsupported_reason)
from B1.metrics import integer_latency, measure_model
from B1.model import build_model
from B1.modules import QuantConv2d, QuantCrossAttention, QuantLinear
from B1.quantizers import ActivationQuantizer, LSQWeightQuantizer
from B1.tests.helpers import CONFIGS, settings, trained_model

KERNEL_ERROR = integer_kernel_error()
needs_kernel = pytest.mark.skipif(KERNEL_ERROR is not None, reason=str(KERNEL_ERROR))


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


@needs_kernel
@pytest.mark.parametrize("name", CONFIGS)
def test_layers_use_the_same_integer_codes_as_fake_quant(tmp_path, name):
    args = settings(tmp_path, *CONFIGS[name])
    model = trained_model(args)
    converted = to_integer(model)
    images = torch.randn(3, 3, 64, 64)
    checked = []

    def compare(quantizer, x, fake_output, integer_layer, layer_name):
        # The fake-quant tensor is x_hat = step * code + offset: recover the codes it used.
        offset = 0 if quantizer.offset is None else quantizer.offset
        fake_codes = ((quantizer(x) - offset) / quantizer.step).round()
        assert torch.equal(fake_codes, integer_layer.quantize(x).float() - integer_layer.zero), layer_name
        tolerance = 2e-5 * fake_output.abs().max().item()
        assert torch.allclose(integer_layer(x), fake_output, rtol=0, atol=tolerance), layer_name
        checked.append(layer_name)

    def hook(layer_name):
        def run(module, inputs, output):
            compare(module.input_quant, inputs[0], output, converted.get_submodule(layer_name), layer_name)
        return run

    handles = [module.register_forward_hook(hook(layer_name)) for layer_name, module in model.named_modules()
               if isinstance(module, (QuantLinear, QuantConv2d))]
    with torch.inference_mode():
        model(images)
        for handle in handles:
            handle.remove()
        x = model.forward_embed(images)
        for index, block in enumerate(model.blocks[:-1]):  # Scorer inputs are the tokens after each block.
            x = block(x)
            if not model.use_cropr or index >= len(model.cropr):
                continue
            source, target = model.cropr[index].cross_attn, converted.cropr[index].cross_attn
            if isinstance(source, QuantCrossAttention):
                compare(source.input_quant, x, source.forward_scorer(x).unsqueeze(-1), target.integer_scorer,
                        f"cropr.{index}")
                assert isinstance(target, IntegerCrossAttention)
            x, _, _ = model.cropr[index](x, inference=True)
    scorers = sum(isinstance(p.cross_attn, QuantCrossAttention) for p in getattr(model, "cropr", []))
    expected = len([m for m in model.modules() if isinstance(m, (QuantLinear, QuantConv2d))]) + scorers
    assert len(checked) == expected


@needs_kernel
@pytest.mark.parametrize("extra,expected_kernels", [([], 48 + 2 + 11), (["--cropr-llf"], 48 + 2 + 10),
                                                    (["--no-use-cropr"], 48 + 2),
                                                    (["--first-last-bits", "32", "--cropr-scorer-bits", "32"], 48)])
def test_converted_model_runs_only_integer_kernels(tmp_path, monkeypatch, extra, expected_kernels):
    """The measured latency must come from integer kernels: no fake quantizer may run in the converted model."""
    converted = to_integer(trained_model(settings(tmp_path, *extra)))
    kernel_calls, fake_quant_calls = [], []
    real_kernel = integer._kernel
    monkeypatch.setattr(integer, "_kernel", lambda *args: kernel_calls.append(args[0].dtype) or real_kernel(*args))
    for quantizer in (ActivationQuantizer, LSQWeightQuantizer):
        forward = quantizer.forward
        monkeypatch.setattr(quantizer, "forward",
                            lambda self, x, forward=forward: fake_quant_calls.append(type(self)) or forward(self, x))
    with torch.inference_mode():
        converted(torch.randn(2, 3, 64, 64))
    assert len(kernel_calls) == expected_kernels
    assert set(kernel_calls) == {torch.uint8} and not fake_quant_calls


@needs_kernel
@pytest.mark.parametrize("bits", [(2, 2), (4, 8), (8, 8)])
def test_whole_model_stays_close_to_fake_quant(tmp_path, bits):
    # Rounding between layers may flip a code at a boundary, so logits are close but not bit-identical.
    args = settings(tmp_path, "--weight-bits", str(bits[0]), "--activation-bits", str(bits[1]))
    model = trained_model(args, seed=1)
    images = torch.randn(32, 3, 64, 64)
    with torch.inference_mode():
        fake, integer_logits = model(images), to_integer(model)(images)
    assert integer_logits.shape == fake.shape and torch.isfinite(integer_logits).all()
    assert (fake.argmax(1) == integer_logits.argmax(1)).float().mean() >= 0.8
    assert (fake - integer_logits).abs().max() < 0.5 * fake.abs().max()


@needs_kernel
def test_conversion_leaves_the_source_model_unchanged_and_only_replaces_quantized_layers(tmp_path):
    model = trained_model(settings(tmp_path))
    before = copy.deepcopy(model.state_dict())
    converted = to_integer(model)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert isinstance(model.blocks[0].mlp.fc1, QuantLinear) and not model.training
    assert isinstance(converted.blocks[0].mlp.fc1, IntegerLinear)
    assert isinstance(converted.patch_embed.proj, IntegerPatchEmbed)
    assert not isinstance(converted.blocks[0].norm1, IntegerMatmul)
    assert type(converted.cropr[0].cross_attn.mlp.fc1) is nn.Linear  # Training-only branch untouched.
    with pytest.raises(RuntimeError, match="inference only"):
        converted.cropr[0].cross_attn(torch.randn(1, 5, 192))
    with torch.inference_mode():
        assert torch.equal(model(torch.zeros(1, 3, 64, 64)), model(torch.zeros(1, 3, 64, 64)))


@needs_kernel
def test_unsupported_states_are_rejected_not_silently_converted(tmp_path):
    model = trained_model(settings(tmp_path))
    quantizer = model.blocks[0].mlp.fc1.input_quant
    quantizer.step.data.fill_(-0.01)
    with pytest.raises(ValueError, match="positive activation step"):
        to_integer(model)
    fresh = build_model(settings(tmp_path)).eval()  # Activation ranges never initialized by a training batch.
    with pytest.raises(ValueError, match="initialized"):
        IntegerLinear(fresh.blocks[0].mlp.fc1)
    conv = model.patch_embed.proj
    conv.stride = (8, 8)
    with pytest.raises(ValueError, match="kernel == stride"):
        IntegerPatchEmbed(conv)
    statsq = trained_model(settings(tmp_path, "--weight-quantizer", "statsq"))
    assert "half-integers" in unsupported_reason(statsq)
    with pytest.raises(RuntimeError, match="half-integers"):
        to_integer(statsq)
    fields = integer_latency(statsq, SimpleNamespace(device="cpu", image_size=64, latency_warmup=0, latency_repeats=1))
    assert fields["integer_latency_ms"] is None and "half-integers" in fields["integer_latency_note"]


def test_inexact_kernel_is_detected_and_reported_instead_of_measured(tmp_path, monkeypatch):
    # A kernel that saturates (as u8s8 without VNNI) returns wrong values without raising.
    real = integer._kernel
    monkeypatch.setattr(integer, "_kernel", lambda *args: real(*args) * 0.5)
    integer.integer_kernel_error.cache_clear()
    try:
        assert integer_kernel_error() is not None
        model = trained_model(settings(tmp_path))
        fields = integer_latency(model, SimpleNamespace(device="cpu", image_size=64, latency_warmup=0, latency_repeats=1))
        assert fields["integer_latency_ms"] is None and fields["integer_latency_note"]
        with pytest.raises(RuntimeError):
            to_integer(model)
    finally:
        integer.integer_kernel_error.cache_clear()


@pytest.mark.skipif(torch.cuda.is_available(), reason="Checks the message on machines without a CUDA GPU")
def test_cuda_evaluation_uses_tensorrt_and_explains_when_it_cannot(tmp_path):
    model = trained_model(settings(tmp_path))
    args = SimpleNamespace(device="cuda", image_size=64, latency_warmup=0, latency_repeats=1, output=str(tmp_path))
    fields = integer_latency(model, args)
    assert fields["integer_latency_ms"] is None and "TensorRT" in fields["integer_latency_note"]


@needs_kernel
def test_metrics_report_integer_latency_next_to_fake_quant_latency(tmp_path):
    args = settings(tmp_path)
    model = trained_model(args)
    metrics = measure_model(model, args)
    assert metrics["integer_latency_ms"] > 0 and metrics["integer_latency_ms_p95"] >= metrics["integer_latency_ms"]
    assert metrics["integer_latency_backend"] == BACKEND and "INT8" in metrics["integer_latency_note"]
    assert metrics["latency_backend"] == "fake_quant_fp32" and metrics["latency_ms_median"] > 0
    assert metrics["collapsed_quantizer_channels"] == 0
    assert isinstance(model.blocks[0].mlp.fc1, QuantLinear)  # Measuring does not convert the evaluated model.
    assert all(not q.initialized.requires_grad for q in model.modules() if isinstance(q, ActivationQuantizer))
