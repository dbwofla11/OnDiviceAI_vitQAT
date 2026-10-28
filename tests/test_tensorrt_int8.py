"""CPU numerical/export checks; these do not establish native TensorRT speed."""
import copy
import csv
from types import SimpleNamespace

import pytest
import torch

from vitqat.quantization import QATLinear, Quantizer
from vitqat.tensorrt_int8 import FrozenQDQ, audit_attention_qdq, convert_tensorrt_int8, export_onnx


def test_w2_values_preserved_and_source_untouched():
    layer = QATLinear(8, 4)
    layer.input_quant = Quantizer(8)
    layer.weight_quant = Quantizer(2, weight=True)
    layer.output_quant = Quantizer(8)
    layer(torch.randn(3, 8) + 2)
    layer.eval()
    expected = layer.weight_quant(layer.weight).detach()
    before = copy.deepcopy(layer.state_dict())
    converted = convert_tensorrt_int8(layer)
    torch.testing.assert_close(converted.weight_quant(converted.weight), expected, rtol=0, atol=0)
    assert all(torch.unique(row).numel() <= 4 for row in converted.weight)
    assert converted.input_quant.zero.item() == 0
    fq = layer.input_quant.fake_quant
    expected_scale = fq.scale * torch.maximum(fq.zero_point, 255 - fq.zero_point) / 127
    torch.testing.assert_close(converted.input_quant.scale, expected_scale)
    assert all(torch.equal(value, layer.state_dict()[key]) for key, value in before.items())


def test_reject_uncalibrated_or_wrong_bits():
    layer = QATLinear(8, 4)
    layer.input_quant = Quantizer(8)
    layer.weight_quant = Quantizer(2, weight=True)
    layer.output_quant = Quantizer(8)
    with pytest.raises(ValueError, match="calibrated"):
        convert_tensorrt_int8(layer)
    layer(torch.randn(3, 8))
    layer.weight_quant.bits = 4
    with pytest.raises(ValueError, match="calibrated"):
        convert_tensorrt_int8(layer)


def test_native_result_summary():
    from scripts.benchmark_tensorrt_int8 import summarize_results

    logits = torch.tensor([[0.25, 0.75]])
    result = summarize_results([1, 4, 3, 2], logits, logits)
    assert result["gpu_compute_ms_median"] == 2.5
    assert result["gpu_compute_ms_p95"] == 4
    assert result["synthetic_native_vs_reference_max_abs_error"] == 0
    with pytest.raises(ValueError, match="timings"):
        summarize_results([], logits, logits)
    with pytest.raises(ValueError, match="logits"):
        summarize_results([1], logits.flatten(), logits)


def test_native_requires_cuda(monkeypatch):
    from scripts.benchmark_tensorrt_int8 import require_tensorrt

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_tensorrt("cuda")


def test_accuracy_keeps_partial_batch_and_uses_sample_weighted_loss():
    from scripts.benchmark_tensorrt_int8 import evaluate_fixed_batch
    from torch.utils.data import DataLoader, TensorDataset

    logits = torch.tensor([[4., 0.], [0., 4.], [1., 0.], [0., 1.], [3., 0.], [0., 3.], [1., 0.]])
    labels = torch.tensor([0, 1, 0, 1, 1, 1, 1])
    seen = []

    def predict(padded):
        seen.append(padded.clone())
        return padded

    batches = DataLoader(TensorDataset(logits, labels), batch_size=4)
    result = evaluate_fixed_batch(predict, batches, 4)
    assert result["evaluated_samples"] == 7
    assert result["top1_acc"] == pytest.approx(100 * 5 / 7)
    assert result["loss"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert len(seen) == 2 and all(batch.shape == (4, 2) for batch in seen)
    assert torch.equal(seen[-1][-1], torch.zeros(2))
    with pytest.raises(ValueError, match="Empty"):
        evaluate_fixed_batch(predict, [], 4)


def test_real_food_loader_and_split_validation(food_fixture):
    from scripts.benchmark_tensorrt_int8 import evaluation_data
    from vitqat.cli import parser
    from vitqat.data import make_data

    config = parser().parse_args(["--mode", "qat", "--output", "unused", "--image-size", "64",
                                 "--data-root", str(food_fixture), "--device", "cpu"])
    _, _, _, manifest = make_data(config)
    options = SimpleNamespace(data_root=str(food_fixture), download=False, batch_size=1, eval_batch_size=32,
                              workers=0, device="cpu", eval_split="test")
    batches, restored = evaluation_data(config, {"split_hash": manifest["sha256"]}, options)
    assert restored == manifest
    assert len(batches.dataset) == 101
    assert [len(target) for _, target in batches] == [32, 32, 32, 5]
    assert next(iter(batches))[0].shape == (32, 3, 64, 64)
    with pytest.raises(ValueError, match="split"):
        evaluation_data(config, {"split_hash": "wrong"}, options)
    with pytest.raises(ValueError, match="split"):
        evaluation_data(config, {}, options)


def test_timing_has_exact_repeat_count(monkeypatch):
    from scripts.benchmark_tensorrt_int8 import time_forwards

    launches, measurements = [], []
    event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None,
                            elapsed_time=lambda end: measurements.append(1.0) or 1.0)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    times = time_forwards(lambda: launches.append(1), SimpleNamespace(synchronize=lambda: None), 3, 7)
    assert len(launches) == 1 + 3 + 7  # initial execution + warmup + timed iterations
    assert times == [1.0] * 7 and len(measurements) == 7


@pytest.mark.parametrize("eval_batch_size", [1, 4])
def test_separate_batch_routing_does_not_replace_timing(monkeypatch, tmp_path, eval_batch_size):
    from scripts import benchmark_tensorrt_int8 as runner

    calls, exports = [], []
    batches = object()

    def engine(trt, output, name, x, reference, options, data=None, *, measure=True):
        calls.append((name, x.shape[0], data, measure))
        result = dict(storage_engine_bytes=100 * x.shape[0], storage_onnx_bytes=50 * x.shape[0])
        if measure:
            result.update(gpu_compute_ms_median=1.5, samples=7)
        if data is not None:
            result.update(top1_acc=75.0, loss=0.5, evaluated_samples=11)
        return result

    monkeypatch.setattr(runner, "benchmark_engine", engine)
    monkeypatch.setattr(runner, "export_onnx", lambda model, x, path: exports.append(x.shape[0]))
    result = runner.run_backend(None, tmp_path, "w2_int8", None, torch.zeros(1, 3, 8, 8),
                                torch.zeros(1, 101), SimpleNamespace(eval_batch_size=eval_batch_size), batches)
    assert result["timing_batch_size"] == 1 and result["eval_batch_size"] == eval_batch_size
    assert result["samples"] == 7 and result["gpu_compute_ms_median"] == 1.5
    assert result["storage_engine_bytes"] == 100
    assert result["eval_storage_engine_bytes"] == 100 * eval_batch_size
    assert result["evaluated_samples"] == 11 and result["top1_acc"] == 75.0
    if eval_batch_size == 1:
        assert calls == [("w2_int8", 1, batches, True)] and not exports
    else:
        assert calls == [("w2_int8", 1, None, True), ("w2_int8_eval_b4", 4, batches, False)]
        assert exports == [4]


def test_csv_keeps_accuracy_and_timing_separate(tmp_path):
    from scripts.benchmark_tensorrt_int8 import write_metrics_csv

    report = dict(eval_split="test", model_structure=dict(params=100, gflops=0.1, tokens_per_block=[17, 8]),
                  source_qat=dict(top1_acc=80.0, evaluated_samples=101),
                  w2_int8=dict(top1_acc=79.0, evaluated_samples=101, gpu_compute_ms_median=2.0))
    write_metrics_csv(tmp_path, report)
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["gpu_compute_ms_median"] == ""
    assert rows[1]["top1_acc"] == "79.0" and rows[1]["evaluated_samples"] == "101"
    assert rows[1]["tokens_per_block"] == "[17, 8]"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native TensorRT requires CUDA PyTorch")
@pytest.mark.parametrize("quantized", [False, True])
def test_native_python_execution(tmp_path, quantized, monkeypatch):
    pytest.importorskip("tensorrt")
    from scripts.benchmark_tensorrt_int8 import run_backend, require_tensorrt
    from scripts import benchmark_tensorrt_int8 as runner

    original_time_forwards = runner.time_forwards
    caller_stream = torch.cuda.current_stream()
    measured_streams = []

    def check_stream(launch, stream, warmup, iterations):
        assert stream == torch.cuda.current_stream()
        assert stream != torch.cuda.default_stream()
        measured_streams.append(stream.cuda_stream)
        return original_time_forwards(launch, stream, warmup, iterations)

    monkeypatch.setattr(runner, "time_forwards", check_stream)

    torch.manual_seed(42)
    x = torch.randn(16, 32)
    if quantized:
        layer = QATLinear(32, 32)
        layer.input_quant, layer.weight_quant, layer.output_quant = Quantizer(8), Quantizer(2, weight=True), Quantizer(8)
        with torch.no_grad():
            layer(x)
        layer = convert_tensorrt_int8(layer.eval())
        tolerance = float(layer.output_quant.scale) * 2
    else:
        layer = torch.nn.Linear(32, 32).eval()
        tolerance = 1e-4
    with torch.no_grad():
        reference = layer(x)
    export_onnx(layer, x[:1], tmp_path / "native.onnx")
    options = SimpleNamespace(device="cuda", warmup=0, iterations=2, eval_batch_size=4)
    from torch.utils.data import DataLoader, TensorDataset
    batches = DataLoader(TensorDataset(x[:7], reference[:7].argmax(-1)), batch_size=4)
    result = run_backend(require_tensorrt("cuda"), tmp_path, "native", layer, x[:1], reference[:1], options, batches)
    assert torch.cuda.current_stream() == caller_stream
    assert len(measured_streams) == 1
    assert result["cuda_stream"] == "dedicated_non_default"
    assert result["samples"] == 2
    assert result["synthetic_native_vs_reference_max_abs_error"] <= tolerance
    assert (tmp_path / "native.engine").stat().st_size > 0
    assert (tmp_path / "native.layers.json").is_file()
    assert result["evaluated_samples"] == 7 and result["top1_acc"] == 100
    assert result["storage_engine_bytes"] > 0
    assert result["timing_batch_size"] == 1 and result["eval_batch_size"] == 4
    assert (tmp_path / "native_eval_b4.engine").is_file()
    assert not (tmp_path / "native_eval_b4.times.json").exists()


@pytest.mark.parametrize("cropr,llf,scope", [
    (False, False, "all"), (True, False, "all"), (True, True, "all"),
    (True, False, "backbone"), (True, True, "backbone"),
])
def test_full_export_and_onnx_runtime(tmp_path, cropr, llf, scope):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from vitqat.cli import parser
    from vitqat.models import build_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = parser().parse_args(["--mode", "qat", "--activation-bits", "8", "--image-size", "64", "--output", "unused"])
    args.use_cropr, args.cropr_llf, args.qat_scope, args.cropr_pruning_rate = cropr, llf, scope, 1
    source = build_model(args)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        source(x)  # Populate all observers, including auxiliary training branches.
    source.eval()
    converted = convert_tensorrt_int8(source)
    assert not any(isinstance(module, Quantizer) for module in converted.modules())
    for name, module in source.named_modules():
        if isinstance(module, Quantizer) and name.endswith(("weight_quant", "query_quant")):
            parent_name, _, role = name.rpartition(".")
            original, target = source.get_submodule(parent_name), converted.get_submodule(parent_name)
            parameter = "weight" if role == "weight_quant" else "queries"
            with torch.no_grad():
                expected = module(getattr(original, parameter))
            torch.testing.assert_close(getattr(target, role)(getattr(target, parameter)), expected, rtol=0, atol=0)
    assert isinstance(converted.blocks[0].attn.prob_quant, FrozenQDQ)
    path = tmp_path / "model.onnx"
    graph = export_onnx(converted, x, path)
    rows = audit_attention_qdq(graph)
    expected_count = 24 + (len(source.cropr) if cropr and scope == "all" else 0)
    assert sum(row["both_inputs_qdq"] for row in rows) == expected_count, rows
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    result = torch.from_numpy(session.run(None, {"images": x.numpy()})[0])
    with torch.no_grad():
        expected = converted(x)
    # Small GEMM/LayerNorm differences can cross quantization boundaries and
    # accumulate across 12 blocks. This is a drift bound, not bitwise equivalence.
    torch.testing.assert_close(result, expected, atol=float(converted.head.output_quant.scale) * 8, rtol=0)
    assert (result - expected).norm() / expected.norm().clamp_min(1e-6) < 0.05
    if not cropr:
        from scripts.benchmark_tensorrt_int8 import main

        checkpoint = tmp_path / "checkpoint.pt"
        torch.save({"model": source.state_dict(), "config": vars(args)}, checkpoint)
        destination = tmp_path / "cli-export"
        report = main(["--checkpoint", str(checkpoint), "--output", str(destination), "--export-only"])
        assert report["engine_execution"] == "not_run"
        assert report["accuracy_evaluation"] == "not_requested"
        assert report["model_structure"]["gflops"] > 0
        assert report["model_structure"]["params"] == sum(p.numel() for p in source.parameters())
        assert len(report["attention_onnx"]) == 24
        assert (destination / "fp32_master.onnx").is_file()
        assert (destination / "images.bin").stat().st_size == x.numel() * 4
