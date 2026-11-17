"""End-to-end run through utils.runner and the real torchvision Food101 loader on a fixture."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from B1.config import CHECKPOINT_DEFAULTS, parser
from B1.inference import main as infer, parser as inference_parser
from B1.metrics import measure_model
from B1.model import build_model, restore_model
from B1.train import main, train_epoch
from B2.config import parser as b2_parser
from utils.checkpoint import read_checkpoint, save_checkpoint
from utils.reproducibility import seed_all


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def settings(output, *extra):
    return parser().parse_args([
        "--image-size", "64", "--cropr-pruning-rate", "1", "--device", "cpu", "--epochs", "1",
        "--warmup-epochs", "0", "--batch-size", "64", "--workers", "0", "--drop-path", "0",
        "--latency-warmup", "0", "--latency-repeats", "1", "--output", str(output), *extra])


@pytest.fixture
def food_fixture(tmp_path):
    root = tmp_path / "food-101"
    (root / "meta").mkdir(parents=True)
    manifests = {"train": {}, "test": {}}
    for c in range(101):
        name = f"class_{c:03d}"
        (root / "images" / name).mkdir(parents=True)
        for split, count in [("train", 3), ("test", 1)]:
            manifests[split][name] = []
            for i in range(count):
                rel = f"{name}/{split}_{i}"
                Image.new("RGB", (72, 72), (c * 2, 30 + i * 40, 80)).save(root / "images" / f"{rel}.jpg")
                manifests[split][name].append(rel)
    for split, manifest in manifests.items():
        (root / "meta" / f"{split}.json").write_text(json.dumps(manifest))
    return tmp_path


def test_end_to_end_resume_and_reevaluation_on_food101_fixture(food_fixture):
    args = settings(food_fixture / "run", "--data-root", str(food_fixture))
    metrics = main(args)
    out = Path(args.output)
    assert metrics["experiment"] == "cropr_linear_qat" and metrics["mode"] == "linear_qat"
    assert metrics["test_samples"] == 101 and math.isfinite(metrics["test_loss"])
    assert (metrics["weight_bits"], metrics["activation_bits"], metrics["first_last_bits"]) == (2, 2, 8)
    assert metrics["token_pruning_ratio"] == 12 / 16
    assert metrics["storage_model_state_bytes"] == (out / "model_state.pt").stat().st_size
    for name in ("config.json", "split.json", "epochs.jsonl", "last.pt", "best.pt", "metrics.csv"):
        assert (out / name).exists(), name
    state = torch.load(out / "best.pt", weights_only=True)["model"]
    assert state["blocks.0.mlp.fc2.input_quant.initialized"].item()
    assert "blocks.0.mlp.fc2.weight_quant.step" in state and "cropr.0.cross_attn.query_quant.step" in state
    args.resume = str(out / "last.pt")
    assert main(args)["top1_acc"] == metrics["top1_acc"]
    # The inference entry point restores bits/quantizers from the checkpoint config.
    evaluated = infer(inference_parser().parse_args([
        "--eval-checkpoint", str(out / "best.pt"), "--output", str(food_fixture / "eval"),
        "--data-root", str(food_fixture), "--device", "cpu", "--workers", "0", "--batch-size", "64",
        "--latency-warmup", "0", "--latency-repeats", "1"]))
    assert evaluated["top1_acc"] == metrics["top1_acc"] and evaluated["test_loss"] == metrics["test_loss"]
    assert evaluated["weight_quantizer"] == "lsq+" and evaluated["activation_quantizer"] == "lsq+"


def test_checkpoint_resume_reproduces_training(tmp_path):
    args = settings(tmp_path / "run", "--batch-size", "2", "--accum-steps", "2")
    seed_all(5)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = DataLoader(TensorDataset(torch.randn(5, 3, 64, 64), torch.arange(5)), batch_size=2)
    train_epoch(model, batches, optimizer, args, 0, 2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0, 1.0, args, "split")
    train_epoch(model, batches, optimizer, args, 1, 2)
    expected = copy.deepcopy(model.state_dict())
    other = build_model(args)
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=args.lr)
    restore = dict(config_defaults=CHECKPOINT_DEFAULTS, restore_model=restore_model)
    read_checkpoint(path, other, args, "split", other_optimizer, **restore)
    train_epoch(other, batches, other_optimizer, args, 1, 2)
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in expected.items())
    args.weight_bits = 4
    with pytest.raises(ValueError, match="weight_bits"):
        read_checkpoint(path, other, args, "split", **restore)


@pytest.mark.parametrize("extra,message", [
    (["--init-checkpoint", "missing.pt"], "not found"),
    (["--pretrained", "--init-checkpoint", "missing.pt"], "either"),
    (["--aux-loss-weight", "0"], "auxiliary"),
])
def test_invalid_options_rejected_before_creating_run(tmp_path, extra, message):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        main(settings(out, *extra))
    assert not out.exists()


def test_shared_training_defaults_match_b2():
    ours = vars(parser().parse_args(["--output", "x"]))
    theirs = vars(b2_parser().parse_args(["--output", "x"]))
    shared = set(ours) & set(theirs) - {"mode"}
    assert {key: ours[key] for key in shared} == {key: theirs[key] for key in shared}
    assert ours["use_cropr"] and (ours["weight_bits"], ours["activation_bits"]) == (2, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_224_training_step_and_metrics(tmp_path):
    args = settings(tmp_path / "cuda", "--image-size", "224", "--cropr-pruning-rate", "8", "--device", "cuda")
    model = build_model(args).cuda()
    batches = DataLoader(TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1])), batch_size=2)
    stats = train_epoch(model, batches, torch.optim.AdamW(model.parameters(), lr=args.lr), args, 0, 1)
    assert stats["train_aux_loss"] > 0
    metrics = measure_model(model, args)
    assert metrics["token_pruning_ratio"] == 89 / 196 and metrics["latency_ms_mean"] > 0
