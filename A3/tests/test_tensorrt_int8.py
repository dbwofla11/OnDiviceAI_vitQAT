"""A3 export integration; synthetic fixtures do not establish task accuracy."""
import copy

import numpy as np
import pytest
import torch

from A3.tests.test_inference_pruning import settings, small_fixture_model


@pytest.mark.parametrize("trained_pruning", [False, True])
def test_paired_export(tmp_path, monkeypatch, trained_pruning):
    ort = pytest.importorskip("onnxruntime")
    from A3 import benchmark_tensorrt_int8 as runner
    from A3.pruning import attach_pruning
    from vitqat.tensorrt_int8 import convert_tensorrt_int8, export_onnx

    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = settings(tmp_path)
    config.weight_bits = 2
    model = small_fixture_model(config)
    rate = 8 if trained_pruning else 16
    if trained_pruning:
        config.cls_pruning = "train_and_inference"
        config.cls_pruning_rate = rate
        model = attach_pruning(model, pruning_rate=rate, apply_during=config.cls_pruning)
    model(torch.randn(2, 3, 224, 224))
    model.eval()
    state = copy.deepcopy(model.state_dict())
    checkpoint = tmp_path / "a2.pt"
    torch.save(dict(config=vars(config), model=state), checkpoint)
    monkeypatch.setattr(runner, "build_model", small_fixture_model)
    output = tmp_path / "export"
    report = runner.main(["--checkpoint", str(checkpoint), "--output", str(output), "--export-only"])
    assert report["engine_execution"] == "not_run"
    assert report["kernel_precision_verified"] is False
    assert report["results"]["a3_int8"]["tokens_per_block"] == [197] + [196 - rate * i for i in range(1, 11)] + [197]
    assert report["results"]["dense_int8"]["tokens_per_block"] == [197] * 12
    assert report["results"]["a3_int8"]["gflops"] < report["results"]["dense_int8"]["gflops"]
    saved = torch.load(checkpoint, weights_only=True)["model"]
    assert all(torch.equal(v, saved[k]) for k, v in state.items())
    converted = convert_tensorrt_int8(model.backbone if trained_pruning else model)
    models = {"dense_int8": converted, "a3_int8": attach_pruning(converted, pruning_rate=rate).eval()}
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    for name, net in models.items():
        session = ort.InferenceSession(str(output / f"{name}.onnx"), sess_options=session_options,
                                       providers=["CPUExecutionProvider"])
        # Multiple actual inputs ensure selection was traced as data-dependent operations.
        for seed in (7, 11):
            torch.manual_seed(seed)
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                expected = net(x).numpy()
            actual = session.run(None, {"images": x.numpy()})[0]
            assert np.isfinite(actual).all()
            # Numerical errors can change quantization bins and tied token rankings.
            if trained_pruning:
                # Same eight-output-bin bound as the W2A8 export regression:
                # Q/DQ rounding can change tied attention rankings across runtimes.
                np.testing.assert_allclose(actual, expected, atol=float(converted.head.output_quant.scale) * 8, rtol=0)
            else:
                assert np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-6) < 0.05
    x = torch.randn(2, 3, 224, 224)
    path = output / "a3_int8_eval_b2.onnx"
    export_onnx(models["a3_int8"], x, path)
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = models["a3_int8"](x).numpy()
    actual = session.run(None, {"images": x.numpy()})[0]
    assert actual.shape == (2, 101)
    if trained_pruning:
        np.testing.assert_allclose(actual, expected, atol=float(converted.head.output_quant.scale) * 8, rtol=0)
    else:
        assert np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-6) < 0.05
    assert (output / "metrics.csv").is_file()
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", str(checkpoint), "--output", str(output), "--export-only"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA PyTorch required")
def test_native_pair(tmp_path, monkeypatch):
    pytest.importorskip("tensorrt")
    from A3 import benchmark_tensorrt_int8 as runner
    from torch.utils.data import DataLoader, TensorDataset

    config = settings(tmp_path)
    config.weight_bits = 2
    model = small_fixture_model(config)
    images = torch.randn(3, 3, 224, 224)
    model(images)
    checkpoint = tmp_path / "a2.pt"
    torch.save(dict(config=vars(config), model=model.state_dict()), checkpoint)
    batches = DataLoader(TensorDataset(images, torch.zeros(3, dtype=torch.long)), batch_size=2)
    monkeypatch.setattr(runner, "build_model", small_fixture_model)
    monkeypatch.setattr(runner, "evaluation_data", lambda *args: (batches, {"sha256": "synthetic"}))
    report = runner.main(["--checkpoint", str(checkpoint), "--output", str(tmp_path / "native"),
                          "--eval-batch-size", "2", "--warmup", "0", "--iterations", "2"])
    assert report["comparison"]["latency_speedup"] > 0
    for row in report["results"].values():
        assert row["evaluated_samples"] == 3
        assert row["samples"] == 2
