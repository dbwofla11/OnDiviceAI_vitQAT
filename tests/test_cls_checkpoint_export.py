"""Wrapped training checkpoints must retain their inference policy on export."""
import pytest
import torch

from A3.pruning.checkpoint import checkpoint_model
from A3.tests.test_inference_pruning import settings, small_fixture_model


@pytest.mark.parametrize("mode", ["train", "train_and_inference"])
def test_wrapped_checkpoint_export(tmp_path, monkeypatch, mode):
    pytest.importorskip("onnx")
    from scripts import benchmark_tensorrt_int8 as runner

    torch.set_num_threads(2)
    config = settings(tmp_path)
    config.weight_bits = 2
    config.cls_pruning = mode
    config.cls_pruning_rate = 8
    model = checkpoint_model(small_fixture_model(config), config)
    model(torch.randn(2, 3, 224, 224))
    path = tmp_path / "trained.pt"
    torch.save(dict(config=vars(config), model=model.state_dict()), path)
    monkeypatch.setattr(runner, "build_model", small_fixture_model)
    report = runner.main(["--checkpoint", str(path), "--output", str(tmp_path / "export"), "--export-only"])
    expected = [197] * 12 if mode == "train" else [197] + [196 - 8 * i for i in range(1, 11)] + [197]
    assert report["model_structure"]["tokens_per_block"] == expected
    assert len(report["attention_onnx"]) == 24
    assert all(row["both_inputs_qdq"] for row in report["attention_onnx"])
    assert (tmp_path / "export" / "fp32_master.onnx").exists()
    assert (tmp_path / "export" / "w2_int8.onnx").exists()


def test_pytorch_evaluation_restores_wrapper(tmp_path, monkeypatch):
    from torch.utils.data import TensorDataset
    from vitqat import train
    from vitqat.cli import parser
    from vitqat.runner import run

    torch.set_num_threads(2)
    config = settings(tmp_path)
    config.weight_bits = 2
    config.cls_pruning, config.cls_pruning_rate = "train_and_inference", 8
    model = checkpoint_model(small_fixture_model(config), config)
    images = torch.randn(2, 3, 224, 224)
    model(images)
    path = tmp_path / "trained.pt"
    torch.save(dict(config=vars(config), model=model.state_dict(), split_hash="fixture",
                    epoch=0, best_acc=0.0), path)
    data = TensorDataset(images, torch.tensor([0, 1]))
    monkeypatch.setattr(train, "build_model", small_fixture_model)
    monkeypatch.setattr(train, "run", lambda *a, **kw: run(
        *a, **kw, data_factory=lambda _: (data, data, data, {"sha256": "fixture"})))
    args = parser().parse_args(["--eval-checkpoint", str(path), "--output", str(tmp_path / "eval"),
                               "--eval-backend", "pytorch", "--device", "cpu", "--workers", "0",
                               "--latency-warmup", "0", "--latency-repeats", "1"])
    result = train.main(args)
    assert result["tokens_per_block"] == [197] + [196 - 8 * i for i in range(1, 11)] + [197]
    assert result["test_samples"] == 2
