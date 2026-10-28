"""Evaluation restores saved settings before data/model construction."""
import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from vitqat.cli import parser
from vitqat.runner import run


@pytest.mark.parametrize("failure", [None, "split", "weights"])
def test_evaluation_restores_config_and_keeps_runtime_options(tmp_path, monkeypatch, failure):
    torch.set_num_threads(2)
    saved = parser().parse_args(["--mode", "qat", "--output", "old-run",
        "--model", "saved-model", "--image-size", "64", "--activation-bits", "8",
        "--use-cropr", "--cropr-llf", "--cropr-pruning-rate", "1", "--qat-scope", "backbone",
        "--mixup", "0.8", "--cutmix", "1.0", "--seed", "7", "--val-fraction", "0.2",
        "--epochs", "1", "--warmup-epochs", "0", "--batch-size", "128"])
    original = nn.Linear(4, 2)
    path = tmp_path / "best.pt"
    torch.save(dict(config=vars(saved), model=original.state_dict(), split_hash="saved-split",
                    epoch=0, best_acc=50.0), path)
    args = parser().parse_args(["--eval-checkpoint", str(path), "--output", str(tmp_path / "eval"),
        "--device", "cpu", "--data-root", "new-data", "--batch-size", "2", "--workers", "0",
        "--latency-warmup", "0", "--latency-repeats", "1"])
    assert args.mode is None and args.mixup == 0.0
    seen = []

    def check_config(config):
        for key in ("mode", "model", "image_size", "activation_bits", "use_cropr", "cropr_llf",
                    "cropr_pruning_rate", "qat_scope", "seed", "val_fraction", "mixup", "cutmix", "epochs"):
            assert getattr(config, key) == getattr(saved, key), key
        assert config.device == "cpu" and config.data_root == "new-data"
        assert config.batch_size == 2 and config.latency_repeats == 1

    def make_data(config):
        check_config(config)
        seen.append("data")
        data = TensorDataset(torch.randn(2, 4), torch.tensor([0, 1]))
        return data, data, data, {"sha256": "wrong" if failure == "split" else "saved-split"}

    def make_model(config):
        check_config(config)
        seen.append("model")
        return nn.Linear(4, 3 if failure == "weights" else 2)

    def forbidden(*a, **kw):
        pytest.fail("Evaluation must not construct an optimizer or train")

    def measure(model, config):
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.state_dict().items())
        return {}

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    kwargs = dict(build_model=make_model, measure_model=measure, data_factory=make_data, train_epoch_fn=forbidden)
    if failure:
        with pytest.raises(ValueError if failure == "split" else RuntimeError,
                           match="split" if failure == "split" else "size mismatch"):
            run(args, **kwargs)
    else:
        result = run(args, **kwargs)
        assert result["test_samples"] == 2
        effective = json.loads((tmp_path / "eval/config.json").read_text())
        assert effective["mixup"] == 0.8 and effective["batch_size"] == 2
        assert not (tmp_path / "eval/last.pt").exists()
    assert seen == ["data", "model"]
    assert args.mode is None  # Caller arguments are not mutated.


def test_training_still_requires_mode(tmp_path):
    from vitqat.train import main
    args = parser().parse_args(["--output", str(tmp_path)])
    with pytest.raises(ValueError, match="--mode"):
        main(args)
