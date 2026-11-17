"""Experiment defaults, checkpoint restoration and boundaries through real runners."""
import importlib
from functools import partial

import pytest
import torch
from torch.utils.data import TensorDataset

from utils.runner import run as run_training
from utils.evaluation import run as run_evaluation


@pytest.mark.parametrize("name,mode,cropr,pruning", [
    ("A1", "fp32", False, None),
    ("A2", "qat", False, None),
    ("A4", "qat", False, "train_and_inference"),
    ("B1", "linear_qat", True, None),
    ("B2", "qat", True, None),
])
def test_training_and_saved_checkpoint_evaluation(tmp_path, monkeypatch, name, mode, cropr, pruning):
    torch.set_num_threads(2)
    entry = importlib.import_module(f"{name}.train")
    images = torch.randn(2, 3, 64, 64)
    dataset = TensorDataset(images, torch.tensor([0, 1]))

    def data_factory(config):
        assert (config.mode, config.use_cropr, config.cls_pruning) == (mode, cropr, pruning)
        assert config.image_size == 64
        return dataset, dataset, dataset, {"sha256": "entrypoint-fixture"}

    inference = importlib.import_module(f"{name}.inference")
    monkeypatch.setattr(entry, "make_data", data_factory)
    monkeypatch.setattr(inference, "make_data", data_factory)
    runtime = ["--device", "cpu", "--workers", "0", "--batch-size", "2",
               "--latency-warmup", "0", "--latency-repeats", "1"]
    args = entry.parser().parse_args([
        "--output", str(tmp_path / "train"), "--image-size", "64",
        "--epochs", "1", "--warmup-epochs", "0", "--drop-path", "0",
        *(["--cropr-pruning-rate", "1"] if cropr else []),
        *(["--cls-pruning-rate", "1"] if pruning else []), *runtime,
    ])
    trained = entry.main(args)
    expected_label = {"A1": "1-1", "A2": "1-2", "A4": "cls_train_and_inference_qat",
                      "B1": "cropr_linear_qat", "B2": "cropr_qat"}[name]
    assert trained["experiment"] == expected_label
    saved_path = tmp_path / "train/best.pt"
    before = torch.load(saved_path, weights_only=True)
    # Architecture/pruning rates must come from the saved config, not CLI defaults.
    args = inference.parser().parse_args([
        "--eval-checkpoint", str(saved_path), "--output", str(tmp_path / "eval"), *runtime,
    ])
    result = inference.main(args)
    assert result["experiment"] == expected_label
    assert result["top1_acc"] == trained["top1_acc"]
    assert result["test_loss"] == trained["test_loss"]
    assert result["tokens_per_block"] == trained["tokens_per_block"]
    assert (result["token_pruning_ratio"] > 0) == bool(cropr or pruning)
    after = torch.load(saved_path, weights_only=True)
    assert before["config"] == after["config"]
    assert all(torch.equal(value, after["model"][key]) for key, value in before["model"].items())
    assert not (tmp_path / "eval/last.pt").exists()


@pytest.mark.parametrize("name,options", [
    ("A1", ["--mode", "qat"]),
    ("A2", ["--use-cropr"]),
    ("A4", ["--cls-pruning", "inference"]),
    ("B1", ["--mode", "fp32"]),
    ("B2", ["--no-use-cropr"]),
])
def test_conflicting_experiment_options_fail_before_creating_output(tmp_path, name, options):
    entry = importlib.import_module(f"{name}.train")
    output = tmp_path / "invalid"
    with pytest.raises((ValueError, SystemExit)):
        args = entry.parser().parse_args(["--output", str(output), *options])
        entry.main(args)
    assert not output.exists()


def test_evaluation_rejects_checkpoint_from_another_experiment(tmp_path, monkeypatch):
    from A1.train import main, parser

    saved = vars(parser().parse_args(["--output", "unused"]))
    saved["mode"] = "qat"
    path = tmp_path / "qat.pt"
    torch.save({"config": saved}, path)
    args = parser().parse_args(["--output", str(tmp_path / "eval"), "--eval-checkpoint", str(path)])
    with pytest.raises(ValueError, match="A1 requires"):
        main(args)


@pytest.mark.parametrize("name", ["A1", "A2", "A4", "B1", "B2"])
def test_inference_cli_has_no_training_policy_flags(name, tmp_path):
    inference = importlib.import_module(f"{name}.inference")
    train = importlib.import_module(f"{name}.train")
    parser = inference.parser()
    for flag, value in [("--epochs", "1"), ("--lr", "0.1"), ("--mixup", "0.8"),
                        ("--model", "another-model"), ("--resume", "last.pt")]:
        with pytest.raises(SystemExit):
            parser.parse_args(["--output", str(tmp_path), "--eval-checkpoint", "best.pt", flag, value])
    args = train.parser().parse_args(["--output", str(tmp_path), "--epochs", "2", "--mixup", "0.8"])
    assert args.epochs == 2 and args.mixup == 0.8


def test_historical_checkpoint_defaults_are_owned_by_experiment(tmp_path, monkeypatch):
    from A1 import train, inference
    from torch import nn
    import json

    config = vars(train.parser().parse_args(["--output", "old-run"]))
    for key in ("mixup", "cutmix", "random_erasing", "crop_min_scale", "pretrained", "cls_pruning"):
        config.pop(key)
    model = nn.Linear(4, 2)
    checkpoint = tmp_path / "old.pt"
    torch.save(dict(config=config, model=model.state_dict(), epoch=0, best_acc=50.0,
                    split_hash="old-split"), checkpoint)
    dataset = TensorDataset(torch.randn(2, 4), torch.tensor([0, 1]))
    monkeypatch.setattr(inference, "build_model", lambda args: nn.Linear(4, 2))
    monkeypatch.setattr(inference, "measure_model", lambda model, args: {})
    monkeypatch.setattr(inference, "make_data",
                       lambda args: (dataset, dataset, dataset, {"sha256": "old-split"}))
    args = inference.parser().parse_args(["--output", str(tmp_path / "eval"),
                                         "--eval-checkpoint", str(checkpoint), "--device", "cpu", "--workers", "0"])
    inference.main(args)
    effective = json.loads((tmp_path / "eval/config.json").read_text())
    assert effective["mixup"] == effective["cutmix"] == effective["random_erasing"] == 0
    assert effective["crop_min_scale"] == 0.08
    assert effective["pretrained"] is False and effective["cls_pruning"] is None
