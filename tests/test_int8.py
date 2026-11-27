"""Shared INT8 building blocks (utils.int8) and the evaluation-time TensorRT measurement (utils.tensorrt).

TensorRT itself needs a CUDA GPU, so here the engine runner is replaced by a recorder; the ONNX graphs
are checked with ONNX Runtime when it is installed.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from utils import tensorrt as runner
from utils.int8 import (ActivationQDQ, GridQuant, Int8Conv2d, Int8Linear, Int8MatMul, affine_grid,
                        audit_int8_products, export_onnx, int8_products, lsq_grid, weight_codes)


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def calibrated(bits, data, weight=False):
    kwargs = dict(quant_min=0, quant_max=2 ** bits - 1, dtype=torch.quint8)
    if weight:
        kwargs.update(observer=MovingAveragePerChannelMinMaxObserver, qscheme=torch.per_channel_affine, ch_axis=0)
    else:
        kwargs.update(observer=MovingAverageMinMaxObserver, qscheme=torch.per_tensor_affine)
    fake_quant = FakeQuantize(**kwargs)
    fake_quant(data)
    fake_quant.disable_observer()
    return fake_quant


def code_errors(actual, expected, scale):
    """Fraction of values whose code differs, and the largest difference in codes."""
    difference = ((actual - expected) / scale).abs()
    return (difference > 0.5).float().mean().item(), difference.max().item()


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("shift", [0.0, 0.7, -3.0])  # zero point at the edge, inside, beyond the range
def test_affine_activation_grid_reproduces_fake_quantize(bits, shift):
    torch.manual_seed(bits)
    data = torch.randn(4, 50, 64) + shift
    fake_quant = calibrated(bits, data)
    x = torch.randn(4, 50, 64) * 1.3 + shift
    qdq = ActivationQDQ(affine_grid(fake_quant))
    with torch.no_grad():
        expected, exported = fake_quant(x), qdq(x) + qdq.offset
    fraction, largest = code_errors(exported, expected, fake_quant.scale.item())
    assert fraction <= 1e-3 and largest <= 1.01
    signed = qdq(x) / fake_quant.scale.item()  # What QuantizeLinear stores must fit signed INT8.
    assert signed.min() >= -128.001 and signed.max() <= 127.001
    assert torch.allclose(GridQuant(affine_grid(fake_quant))(x), expected, atol=1.01 * fake_quant.scale.item())


@pytest.mark.parametrize("asymmetric", [True, False])
def test_lsq_grid_matches_the_lsq_definition(asymmetric):
    step, offset, bits = 0.05, -0.3, 2
    grid = lsq_grid(step, offset, bits, asymmetric)
    x = torch.linspace(-1, 1, 401)
    if asymmetric:
        expected = ((x - offset) / step).clamp(0, 3).round() * step + offset
    else:
        expected = (x / step).clamp(-2, 1).round() * step
    qdq = ActivationQDQ(grid)
    assert torch.allclose(qdq(x) + qdq.offset, expected, atol=1e-6)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_weight_codes_are_exact_integers_and_shift_only_when_needed(bits):
    torch.manual_seed(bits)
    weight = torch.randn(6, 40) * 0.05 + torch.linspace(-0.2, 0.2, 6)[:, None]  # off-centre channels
    fake_quant = calibrated(bits, weight, weight=True)
    w_hat = fake_quant(weight)
    codes = weight_codes(w_hat, fake_quant.scale, fake_quant.zero_point)
    assert codes.codes.min() >= -128 and codes.codes.max() <= 127
    assert torch.equal(codes.codes, codes.codes.round())
    rebuilt = codes.codes * codes.scale[:, None] + (0 if codes.shift is None else codes.shift[:, None])
    assert torch.allclose(rebuilt, w_hat, atol=1e-6)
    if bits < 8:
        assert codes.shift is None  # q - z already fits INT8.
    with pytest.raises(ValueError, match="not on the given grid"):
        weight_codes(w_hat + 0.3 * fake_quant.scale[:, None], fake_quant.scale, fake_quant.zero_point)


@pytest.mark.parametrize("bits", [(2, 8), (4, 4), (8, 8)])
@pytest.mark.parametrize("with_output", [True, False])
def test_int8_linear_and_conv_compute_the_fake_quant_product(bits, with_output):
    torch.manual_seed(1)
    linear, conv = nn.Linear(24, 10), nn.Conv2d(3, 5, kernel_size=4, stride=4)
    x_linear, x_conv = torch.randn(7, 24) + 0.4, torch.randn(2, 3, 8, 8) + 0.4
    for layer, x in ((linear, x_linear), (conv, x_conv)):
        input_fq = calibrated(bits[1], x)
        weight_fq = calibrated(bits[0], layer.weight.detach(), weight=True)
        output_fq = None
        with torch.no_grad():
            w_hat = weight_fq(layer.weight)
            y = F_apply(layer, input_fq(x), w_hat)
            if with_output:
                output_fq = calibrated(bits[1], y)
                y = output_fq(y)
        grid_out = None if output_fq is None else affine_grid(output_fq)
        codes = weight_codes(w_hat, weight_fq.scale, weight_fq.zero_point)
        int8 = (Int8Linear(affine_grid(input_fq), codes, layer.bias, grid_out) if isinstance(layer, nn.Linear)
                else Int8Conv2d(layer, affine_grid(input_fq), codes, grid_out))
        with torch.no_grad():
            got = int8(x)
        step = grid_out.scale if grid_out else input_fq.scale.item() * w_hat.abs().max().item()
        assert (got - y).abs().max() <= 2.01 * step + 1e-5 * y.abs().max()
        assert (got - y).abs().mean() <= 1e-3 * y.abs().mean() + 1e-6
        assert int8_products(nn.Sequential(int8)) == 1


def F_apply(layer, x_hat, w_hat):
    if isinstance(layer, nn.Linear):
        return nn.functional.linear(x_hat, w_hat, layer.bias)
    return nn.functional.conv2d(x_hat, w_hat, layer.bias, stride=layer.stride, padding=layer.padding)


def test_int8_conv_offset_correction_for_shifted_8_bit_weights():
    torch.manual_seed(3)
    conv = nn.Conv2d(3, 4, kernel_size=2, stride=2)
    with torch.no_grad():
        conv.weight.add_(torch.linspace(0.2, 0.6, 4)[:, None, None, None])  # far from a centred zero point
    weight_fq = calibrated(8, conv.weight.detach(), weight=True)
    x = torch.randn(2, 3, 6, 6) + 1.0
    input_fq = calibrated(8, x)
    with torch.no_grad():
        w_hat = weight_fq(conv.weight)
        codes = weight_codes(w_hat, weight_fq.scale, weight_fq.zero_point)
        assert codes.shift is not None  # the correction path is exercised
        expected = F_apply(conv, input_fq(x), w_hat)
        got = Int8Conv2d(conv, affine_grid(input_fq), codes)(x)
    assert (got - expected).abs().max() <= 1e-4 * expected.abs().max() + 2.01 * input_fq.scale.item() * w_hat.abs().max()
    padded = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    padded_fq = calibrated(8, padded.weight.detach(), weight=True)
    with pytest.raises(ValueError, match="without padding"):  # zeros would not mean x_hat = 0 after the shift
        Int8Conv2d(padded, affine_grid(input_fq), weight_codes(padded_fq(padded.weight), padded_fq.scale,
                                                               padded_fq.zero_point))


@pytest.mark.parametrize("bits", [2, 8])
def test_int8_matmul_of_two_activations_matches_the_fake_quant_product(bits):
    torch.manual_seed(bits)
    a, b = torch.randn(2, 3, 9, 16) + 0.5, torch.randn(2, 3, 16, 7) - 0.2
    a_fq, b_fq = calibrated(bits, a), calibrated(bits, b)
    with torch.no_grad():
        expected = a_fq(a) @ b_fq(b)
        got = Int8MatMul(affine_grid(a_fq), affine_grid(b_fq))(a, b)
    bound = 2.01 * (a_fq.scale * b.abs().max() + b_fq.scale * a.abs().max()).item() + 1e-5 * expected.abs().max()
    assert (got - expected).abs().max() <= bound
    assert (got - expected).abs().mean() <= 1e-3 * expected.abs().mean() + 1e-6


def test_exported_graph_marks_int8_products_and_runs_in_onnxruntime(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(5)
    linear = nn.Linear(16, 8)
    x = torch.randn(1, 4, 16) + 0.2
    input_fq, weight_fq = calibrated(8, x), calibrated(2, linear.weight.detach(), weight=True)
    with torch.no_grad():
        codes = weight_codes(weight_fq(linear.weight), weight_fq.scale, weight_fq.zero_point)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Int8Linear(affine_grid(input_fq), codes, linear.bias)
            self.mm = Int8MatMul(affine_grid(input_fq), affine_grid(input_fq))

        def forward(self, images):
            y = self.layer(images)
            return self.mm(y, y.transpose(-2, -1)).flatten(1)

    model = Model().eval()
    graph = export_onnx(model, x, tmp_path / "model.onnx")
    rows = audit_int8_products(graph)
    assert [row["int8"] for row in rows] == [True, True] and int8_products(model) == 2
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    result = torch.from_numpy(session.run(None, {"images": x.numpy()})[0])
    with torch.no_grad():
        assert torch.allclose(result, model(x), atol=1e-3 * model(x).abs().max().item() + 1e-4)


def test_deployment_latency_reports_why_it_did_not_measure(tmp_path):
    model = nn.Linear(4, 4)
    fields = runner.deployment_latency(model, device="cpu", image_size=8, warmup=0, repeats=1, output=tmp_path,
                                       int8_export=lambda: model)
    assert fields["integer_latency_ms"] is None and "CUDA" in fields["integer_latency_note"]
    fields = runner.deployment_latency(model, device="cpu", image_size=8, warmup=0, repeats=1, output=tmp_path)
    assert fields["tensorrt_fp32_latency_ms"] is None and "CUDA" in fields["tensorrt_fp32_latency_note"]
    assert not (tmp_path / "tensorrt").exists()


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 4, stride=4)
        self.head = nn.Linear(16, 3)

    def forward(self, images):
        return self.head(self.conv(images).flatten(1))


def test_deployment_latency_builds_int8_engines_only_for_int8_exports(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    calls = []

    def fake_engine(trt, output, name, x, reference, options, batches=None, *, evaluate_batches, compare_outputs,
                    allow_tf32, measure=True, int8=False):
        calls.append((name, int8, tuple(x.shape), options.warmup, options.iterations, allow_tf32))
        assert (output / f"{name}.onnx").is_file()
        return dict(gpu_compute_ms_median=2.0, gpu_compute_ms_mean=2.1, gpu_compute_ms_p95=2.5,
                    storage_engine_bytes=123, synthetic_native_vs_reference_max_abs_error=1e-4)

    monkeypatch.setattr(runner, "require_tensorrt", lambda device: SimpleNamespace(__version__="10.13.3"))
    monkeypatch.setattr(runner, "benchmark_engine", fake_engine)
    model = TinyNet().eval()
    fp32 = runner.deployment_latency(model, device="cuda", image_size=8, warmup=3, repeats=7, output=tmp_path,
                                     name="fp32")
    int8 = runner.deployment_latency(model, device="cuda:0", image_size=8, warmup=3, repeats=7, output=tmp_path,
                                     int8_export=lambda: TinyNet().eval(), name="int8")
    assert calls == [("fp32", False, (1, 3, 8, 8), 3, 7, False), ("int8", True, (1, 3, 8, 8), 3, 7, False)]
    assert fp32["tensorrt_fp32_latency_ms"] == 2.0 and fp32["tensorrt_fp32_latency_backend"].endswith("fp32")
    assert int8["integer_latency_ms"] == 2.0 and int8["integer_latency_ms_p95"] == 2.5
    assert int8["integer_latency_backend"] == "tensorrt_10.13.3_int8" and int8["integer_latency_engine_bytes"] == 123
    assert int8["integer_latency_layers_file"].endswith("int8.layers.json")

    def broken(*args, **kwargs):
        raise RuntimeError("engine build failed\nlong log")

    monkeypatch.setattr(runner, "benchmark_engine", broken)
    failed = runner.deployment_latency(model, device="cuda", image_size=8, warmup=0, repeats=1, output=tmp_path,
                                       int8_export=lambda: model, name="broken")
    assert failed["integer_latency_ms"] is None
    assert failed["integer_latency_note"] == "TensorRT measurement failed: RuntimeError: engine build failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native TensorRT requires CUDA PyTorch")
def test_native_engine_runs_affine_int8_blocks_with_offset_corrections(tmp_path):
    """Run on the RTX 4090 machine before a long evaluation: the riskiest TensorRT patterns in one engine.

    An affine Linear with an output grid, a product of two activations with non-zero offsets (row/column-sum
    corrections) and a grid fake-quantization, all exported with zero-point-0 INT8 Q/DQ.
    """
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")
    torch.manual_seed(7)
    linear = nn.Linear(16, 16)
    x = torch.randn(1, 8, 16) + 0.3
    input_fq = calibrated(8, x)
    weight_fq = calibrated(2, linear.weight.detach(), weight=True)
    with torch.no_grad():
        w_hat = weight_fq(linear.weight)
        y = nn.functional.linear(input_fq(x), w_hat, linear.bias)
        output_fq = calibrated(8, y)
    codes = weight_codes(w_hat, weight_fq.scale, weight_fq.zero_point)

    class Blocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Int8Linear(affine_grid(input_fq), codes, linear.bias, affine_grid(output_fq))
            self.mm = Int8MatMul(affine_grid(output_fq), affine_grid(output_fq))

        def forward(self, images):
            y = self.layer(images)
            return self.mm(y, y.transpose(-2, -1)).flatten(1)

    model = Blocks().eval()
    with torch.inference_mode():
        reference = model(x)
    options = SimpleNamespace(device="cuda", warmup=1, iterations=3, eval_batch_size=1)
    result = runner.run_backend(runner.require_tensorrt("cuda"), tmp_path, "affine", model, x, reference, options, None,
                                export_model=export_onnx, evaluate_batches=runner.evaluate_fixed_batch,
                                compare_outputs=runner.compare_logits, allow_tf32=False, int8=True)
    # A flipped code of y changes a product entry by step * |y|; a missing offset correction would be ~|y| * K.
    with torch.no_grad():
        y_max = output_fq(y).abs().max().item()
    tolerance = 4 * output_fq.scale.item() * y_max + 1e-3 * reference.abs().max().item()
    assert result["synthetic_native_vs_reference_max_abs_error"] <= tolerance
    assert result["samples"] == 3 and (tmp_path / "affine.layers.json").is_file()
