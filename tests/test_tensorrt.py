"""Shared TensorRT runner (utils.tensorrt) parts that need no GPU: timing counts, batch routing, accounting."""
from types import SimpleNamespace

import pytest
import torch

from utils.tensorrt import compare_logits, evaluate_fixed_batch


def test_native_result_summary():
    from utils.tensorrt import summarize_results

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
    from utils.tensorrt import require_tensorrt

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_tensorrt("cuda")


def test_accuracy_keeps_partial_batch_and_uses_sample_weighted_loss():
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


def test_timing_has_exact_repeat_count(monkeypatch):
    from utils.tensorrt import time_forwards

    launches, measurements = [], []
    event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None,
                            elapsed_time=lambda end: measurements.append(1.0) or 1.0)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    times = time_forwards(lambda: launches.append(1), SimpleNamespace(synchronize=lambda: None), 3, 7)
    assert len(launches) == 1 + 3 + 7  # initial execution + warmup + timed iterations
    assert times == [1.0] * 7 and len(measurements) == 7


@pytest.mark.parametrize("eval_batch_size", [1, 4])
def test_separate_batch_routing_does_not_replace_timing(monkeypatch, tmp_path, eval_batch_size):
    from utils import tensorrt as runner

    calls, exports = [], []
    batches = object()

    def engine(trt, output, name, x, reference, options, data=None, *, evaluate_batches, compare_outputs, allow_tf32,
               measure=True, int8=False):
        calls.append((name, x.shape[0], data, measure, int8))
        result = dict(storage_engine_bytes=100 * x.shape[0], storage_onnx_bytes=50 * x.shape[0])
        if measure:
            result.update(gpu_compute_ms_median=1.5, samples=7)
        if data is not None:
            result.update(top1_acc=75.0, loss=0.5, evaluated_samples=11)
        return result

    monkeypatch.setattr(runner, "benchmark_engine", engine)
    result = runner.run_backend(None, tmp_path, "w2_int8", None, torch.zeros(1, 3, 8, 8),
                                torch.zeros(1, 101), SimpleNamespace(eval_batch_size=eval_batch_size), batches,
                                export_model=lambda model, x, path: exports.append(x.shape[0]),
                                evaluate_batches=evaluate_fixed_batch, compare_outputs=compare_logits, allow_tf32=False,
                                int8=True)
    assert result["timing_batch_size"] == 1 and result["eval_batch_size"] == eval_batch_size
    assert result["samples"] == 7 and result["gpu_compute_ms_median"] == 1.5
    assert result["storage_engine_bytes"] == 100
    assert result["eval_storage_engine_bytes"] == 100 * eval_batch_size
    assert result["evaluated_samples"] == 11 and result["top1_acc"] == 75.0
    if eval_batch_size == 1:
        assert calls == [("w2_int8", 1, batches, True, True)] and not exports
    else:
        # Both engines of the quantized network request INT8 kernels.
        assert calls == [("w2_int8", 1, None, True, True), ("w2_int8_eval_b4", 4, batches, False, True)]
        assert exports == [4]
