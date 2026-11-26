"""INT8 Q/DQ export for TensorRT: the learned grid is kept and the graph runs in ONNX Runtime.

Nothing here runs TensorRT: the engines need a CUDA GPU, so the layer precision TensorRT picks and the
GPU latency are only known by running B1.benchmark_tensorrt_int8 on the target machine.
"""
import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from B1 import benchmark_tensorrt_int8 as cli
from B1.model import build_backbone, build_model
from B1.modules import QuantConv2d, QuantCrossAttention, QuantLinear, _quantize_layer
from B1.quantizers import ActivationQuantizer
from B1.tensorrt_int8 import (ActivationQDQ, ExportConv2d, ExportCrossAttention, ExportCropr, ExportLinear,
                              audit_quantized_products, convert_for_tensorrt, expected_quantized_products,
                              export_onnx)
from B1.tests.helpers import CONFIGS, settings, trained_model
from B1.tensorrt_engine import compare_logits, evaluate_fixed_batch, require_tensorrt, run_backend


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def fake_codes(quantizer, x_hat):
    offset = 0 if quantizer.offset is None else quantizer.offset
    return ((x_hat - offset) / quantizer.step).round()


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("asymmetric", [True, False])
def test_activation_qdq_keeps_the_learned_grid(bits, asymmetric):
    torch.manual_seed(bits)
    quantizer = ActivationQuantizer(bits, asymmetric=asymmetric)
    x = torch.randn(8, 16, 64) * 2 + (0.3 if asymmetric else 0)
    quantizer.train()(x)  # initialize on the "first batch"
    quantizer.eval()
    qdq = ActivationQDQ(quantizer)
    with torch.no_grad():
        # Q/DQ works on signed INT8 around a shifted zero; adding the real offset back gives x_hat.
        x_hat, exported = quantizer(x), qdq(x) + qdq.offset
    difference = ((exported - x_hat) / quantizer.step).abs()
    assert difference.max() <= 1.01 and (difference > 0.5).float().mean() <= 1e-3
    codes = fake_codes(quantizer, x_hat)
    assert codes.min() >= quantizer.qmin and codes.max() <= quantizer.qmax
    # The codes handed to QuantizeLinear must fit signed INT8, whatever the bit width.
    with torch.no_grad():
        signed = qdq(x) / quantizer.step
    assert signed.min() >= -128 - 1e-3 and signed.max() <= 127 + 1e-3


@pytest.mark.parametrize("name", CONFIGS)
def test_exported_layers_match_fake_quant_layers(tmp_path, name):
    model = trained_model(settings(tmp_path, *CONFIGS[name]))
    exported = convert_for_tensorrt(model)
    images = torch.randn(3, 3, 64, 64)
    seen = []

    def check(module, output, exported_output, weight_hat, layer_name):
        # One activation code may flip at a rounding boundary: it changes an output by step * |w_hat|.
        step = module.input_quant.step.item()
        limit = 4 * step * weight_hat.abs().max().item() + 1e-4 * output.abs().max().item()
        assert (exported_output - output).abs().max().item() <= limit, layer_name
        assert (exported_output - output).abs().mean() <= 1e-3 * output.abs().mean() + 1e-6, layer_name
        seen.append(layer_name)

    def hook(layer_name):
        def run(module, inputs, output):
            check(module, output, exported.get_submodule(layer_name)(inputs[0]),
                  module.weight_quant(module.weight), layer_name)
        return run

    handles = [module.register_forward_hook(hook(layer_name)) for layer_name, module in model.named_modules()
               if isinstance(module, (QuantLinear, QuantConv2d))]
    with torch.inference_mode():
        model(images)
        for handle in handles:
            handle.remove()
        x = model.forward_embed(images)
        for index, block in enumerate(model.blocks[:-1]):
            x = block(x)
            if not model.use_cropr or index >= len(model.cropr):
                continue
            source, target = model.cropr[index].cross_attn, exported.cropr[index].cross_attn
            if isinstance(source, QuantCrossAttention):
                check(source, source.forward_scorer(x).unsqueeze(-1), target.forward_scorer(x).unsqueeze(-1),
                      source.query_quant(source.queries[0]), f"cropr.{index}")
                assert isinstance(target, ExportCrossAttention)
            x, _, _ = model.cropr[index](x, inference=True)
    quantized = len([m for m in model.modules() if isinstance(m, (QuantLinear, QuantConv2d, QuantCrossAttention))])
    assert len(seen) == quantized == expected_quantized_products(exported)


@pytest.mark.parametrize("extra", [[], ["--activation-bits", "8"], ["--weight-bits", "4", "--activation-bits", "8"]])
def test_whole_exported_model_stays_close_to_fake_quant(tmp_path, extra):
    model = trained_model(settings(tmp_path, *extra), seed=1)
    images = torch.randn(24, 3, 64, 64)
    with torch.inference_mode():
        fake, exported = model(images), convert_for_tensorrt(model)(images)
    assert torch.isfinite(exported).all() and (fake - exported).abs().max() < 0.5 * fake.abs().max()
    assert (fake.argmax(1) == exported.argmax(1)).float().mean() >= 0.8


def test_export_leaves_the_source_model_unchanged(tmp_path):
    model = trained_model(settings(tmp_path, "--activation-bits", "8"))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    exported = convert_for_tensorrt(model)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert isinstance(model.blocks[0].mlp.fc1, QuantLinear)
    assert isinstance(exported.blocks[0].mlp.fc1, ExportLinear)
    assert isinstance(exported.patch_embed.proj, ExportConv2d)
    assert isinstance(exported.cropr[0], ExportCropr) and not isinstance(model.cropr[0], ExportCropr)
    with pytest.raises(RuntimeError, match="inference only"):
        exported.cropr[0](torch.randn(1, 5, 192), inference=False)
    with pytest.raises(RuntimeError, match="inference only"):
        exported.cropr[0].cross_attn(torch.randn(1, 5, 192))


def test_unsupported_models_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="half-integers"):
        convert_for_tensorrt(trained_model(settings(tmp_path, "--weight-quantizer", "statsq")))
    with pytest.raises(ValueError, match="initialized"):
        convert_for_tensorrt(build_model(settings(tmp_path)).eval())  # Activation ranges never calibrated.
    fp32_layers_only = trained_model(settings(tmp_path, "--first-last-bits", "32", "--cropr-scorer-bits", "32"))
    assert expected_quantized_products(convert_for_tensorrt(fp32_layers_only)) == 48
    with pytest.raises(ValueError, match="no Linear QAT layers"):
        convert_for_tensorrt(nn.Sequential(nn.Linear(4, 4)))


def run_onnxruntime(path, images):
    session = pytest.importorskip("onnxruntime").InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return torch.from_numpy(session.run(None, {"images": images.numpy()})[0])


@pytest.mark.parametrize("name", ["W2A2", "W2A8", "W8A8", "W2A8 symmetric activation", "LLF", "dense"])
def test_exported_graph_has_int8_products_and_runs_in_onnxruntime(tmp_path, name):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = trained_model(settings(tmp_path, *CONFIGS[name]))
    exported = convert_for_tensorrt(model)
    images = torch.randn(3, 3, 64, 64)
    path = tmp_path / "model.onnx"
    graph = export_onnx(exported, images[:1], path)
    products = audit_quantized_products(graph)
    quantized = sum(row["quantized"] for row in products)
    expected = expected_quantized_products(exported)
    assert quantized == expected
    # Attention QK^T/AV stay FP32: 2 products per block, or none once blocks are gone.
    assert len(products) - quantized == 2 * len(model.blocks)
    ops = [node.op_type for node in graph.graph.node]
    # One Q/DQ pair per input and weight; weights must not be folded into FP32 constants.
    assert ops.count("QuantizeLinear") == ops.count("DequantizeLinear") == 2 * expected
    with torch.inference_mode():
        reference = exported(images)
    results = torch.cat([run_onnxruntime(path, images[i:i + 1]) for i in range(len(images))])
    assert (results - reference).abs().max() < 0.1 * max(1.0, reference.abs().max().item())
    assert (results.argmax(1) == reference.argmax(1)).all()


def test_fp32_reference_model_exports_without_quantization(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = trained_model(settings(tmp_path))
    fp32 = build_backbone(settings(tmp_path))
    fp32.load_state_dict({key: model.state_dict()[key] for key in fp32.state_dict()}, strict=True)
    fp32.eval()
    images = torch.randn(2, 3, 64, 64)
    path = tmp_path / "fp32.onnx"
    graph = export_onnx(fp32, images[:1], path)  # Upstream Cropr uses argsort(stable=False), which is replaced.
    assert not any(row["quantized"] for row in audit_quantized_products(graph))
    with torch.inference_mode():
        reference = fp32(images)
    results = torch.cat([run_onnxruntime(path, images[i:i + 1]) for i in range(2)])
    assert torch.allclose(results, reference, atol=1e-3)
    assert isinstance(fp32.cropr[0], type(model.cropr[0]))  # The exported copy does not touch the source.


def save_checkpoint(path, model, args):
    torch.save(dict(model=model.state_dict(), config=vars(args), split_hash="fixture", epoch=0, best_score=0.0), path)


def test_cli_export_only_writes_both_graphs_and_the_report(tmp_path):
    pytest.importorskip("onnx")
    args = settings(tmp_path, "--activation-bits", "8")
    model = trained_model(args)
    save_checkpoint(tmp_path / "best.pt", model, args)
    report = cli.main(["--checkpoint", str(tmp_path / "best.pt"), "--output", str(tmp_path / "out"), "--export-only"])
    for name in ("fp32_master.onnx", "b1_int8.onnx", "images.bin", "report.json"):
        assert (tmp_path / "out" / name).exists(), name
    assert report["engine_execution"] == "not_run" and report["kernel_precision_verified"] is False
    assert (report["weight_bits"], report["activation_bits"], report["first_last_bits"]) == (2, 8, 8)
    assert report["accuracy_evaluation"] == "not_requested" and report["collapsed_quantizer_channels"] == 0
    assert report["onnx_quantized_products"] == 48 + 2 + 11
    assert report["model_structure"]["gflops"] > 0 and report["synthetic_top1_agreement"] >= 0
    assert json.loads((tmp_path / "out" / "report.json").read_text()) == report


def test_cli_requests_int8_kernels_only_for_the_quantized_engine(tmp_path, monkeypatch):
    """Runs the whole CLI flow with the GPU runner replaced by a recorder."""
    pytest.importorskip("onnx")
    args = settings(tmp_path, "--activation-bits", "8")
    save_checkpoint(tmp_path / "best.pt", trained_model(args), args)
    calls = []

    def fake_run_backend(trt, output, name, model, x, reference, options, batches, *, export_model, evaluate_batches,
                         compare_outputs, allow_tf32, int8=False):
        calls.append((name, int8, type(model).__name__, tuple(x.shape), allow_tf32, reference.shape))
        return dict(gpu_compute_ms_median=4.0 if name == "fp32_master" else 2.0, storage_engine_bytes=10, samples=3)

    class FakeTRT:
        __version__ = "10.13.3"

    monkeypatch.setattr(cli, "require_tensorrt", lambda device: FakeTRT)
    monkeypatch.setattr(cli, "run_backend", fake_run_backend)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake GPU")
    report = cli.main(["--checkpoint", str(tmp_path / "best.pt"), "--output", str(tmp_path / "out"),
                       "--warmup", "1", "--iterations", "3"])
    assert [(call[0], call[1]) for call in calls] == [("fp32_master", False), ("b1_int8", True)]
    assert all(call[3] == (1, 3, 64, 64) and not call[4] and call[5] == (1, 101) for call in calls)
    assert report["gpu_compute_speedup_vs_trt_fp32"] == 2.0 and report["engine_size_ratio_fp32_to_int8"] == 1.0
    assert report["engine_execution"].startswith("completed") and report["backend_version"] == "10.13.3"
    assert (tmp_path / "out" / "metrics.csv").exists() and report["accuracy_evaluation"] == "not_requested"


def test_cli_rejects_bad_options_and_other_experiments(tmp_path, capsys):
    for bad in (["--export-only", "--data-root", "x"], ["--download"], ["--iterations", "0"]):
        with pytest.raises(SystemExit):
            cli.parse_args(["--checkpoint", "x.pt", "--output", "o", *bad])
    capsys.readouterr()
    args = settings(tmp_path)
    model = trained_model(args)
    path = tmp_path / "other.pt"
    args.mode = "qat"
    save_checkpoint(path, model, args)
    with pytest.raises(ValueError, match="linear_qat"):
        cli.load_models(cli.parse_args(["--checkpoint", str(path), "--output", "o", "--export-only"]))


def quantized_linear(weight_bits=2, activation_bits=8):
    torch.manual_seed(0)
    layer = nn.Linear(32, 16)
    _quantize_layer(layer, weight_bits, activation_bits, "lsq+", "lsq+")
    layer.train()(torch.randn(16, 32) + 0.3)  # initialize the activation range
    return layer.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native TensorRT requires CUDA PyTorch")
def test_native_engine_reproduces_the_exported_layer(tmp_path):
    """Smallest end-to-end check of the GPU runner. Run it on the RTX 4090 machine before the benchmark."""
    pytest.importorskip("tensorrt")
    layer = quantized_linear()
    model = nn.Sequential(ExportLinear(layer)).eval()
    x = torch.randn(1, 32)
    with torch.inference_mode():
        reference = model(x)
    options = SimpleNamespace(device="cuda", warmup=1, iterations=3, eval_batch_size=1)
    result = run_backend(require_tensorrt("cuda"), tmp_path, "native", model, x, reference, options, None,
                         export_model=export_onnx, evaluate_batches=evaluate_fixed_batch, compare_outputs=compare_logits,
                         allow_tf32=False, int8=True)
    weight_hat = layer.weight_quant(layer.weight)
    tolerance = 8 * layer.input_quant.step.item() * weight_hat.abs().max().item() + 1e-3
    assert result["synthetic_native_vs_reference_max_abs_error"] <= tolerance
    assert result["samples"] == 3 and result["storage_engine_bytes"] > 0
    assert (tmp_path / "native.layers.json").is_file()


@pytest.mark.skipif(torch.cuda.is_available(), reason="Checks the error on machines without CUDA")
def test_gpu_execution_fails_clearly_without_cuda(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        cli.main(["--checkpoint", "missing.pt", "--output", str(tmp_path / "out")])


def test_metrics_csv_keeps_accuracy_and_timing_of_each_engine(tmp_path):
    report = dict(eval_split="test", model_structure=dict(params=100, gflops=0.1, tokens_per_block=[17, 8]),
                  source_qat=dict(top1_acc=80.0, evaluated_samples=101),
                  b1_int8=dict(top1_acc=79.0, evaluated_samples=101, gpu_compute_ms_median=2.0))
    cli.write_metrics_csv(tmp_path, report)
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["backend"] for row in rows] == ["source_qat", "b1_int8"]
    assert rows[0]["gpu_compute_ms_median"] == "" and rows[1]["top1_acc"] == "79.0"
    assert rows[1]["tokens_per_block"] == "[17, 8]"
