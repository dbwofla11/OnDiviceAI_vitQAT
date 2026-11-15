"""Real experiment pipelines on synthetic inputs; not Food-101 accuracy reproduction."""
import importlib
import json

import pytest
import torch
from torch.utils.data import TensorDataset


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def synthetic_data(size):
    images = torch.randn(3, 3, size, size, generator=torch.Generator().manual_seed(123))
    dataset = TensorDataset(images, torch.tensor([0, 1, 2]))
    return lambda args: (dataset, dataset, dataset, {"sha256": "reproducibility-fixture"})


@pytest.mark.parametrize("name", ["A1", "A2", "A4", "B1", "B2"])
def test_seed_repeat_and_interrupted_resume_match_full_training(tmp_path, monkeypatch, name):
    entry = importlib.import_module(f"{name}.train")
    inference = importlib.import_module(f"{name}.inference")
    data = synthetic_data(64)
    monkeypatch.setattr(entry, "make_data", data)
    monkeypatch.setattr(inference, "make_data", data)
    runtime = ["--device", "cpu", "--workers", "0", "--threads", "2", "--batch-size", "2",
               "--latency-warmup", "0", "--latency-repeats", "1"]
    options = ["--image-size", "64", "--epochs", "2", "--warmup-epochs", "0",
               "--seed", "42", "--drop-path", "0.1", "--mixup", "0.8", "--accum-steps", "2", *runtime]
    if name in {"A2", "A4", "B2"}:
        options += ["--freeze-observers-epoch", "1"]
    if name in {"B1", "B2"}:
        options += ["--cropr-pruning-rate", "1"]
    if name == "A4":
        options += ["--cls-pruning-rate", "1"]

    def train(directory, *extra):
        return entry.main(entry.parser().parse_args(["--output", str(directory), *options, *extra]))

    full = tmp_path / "full"
    repeat = tmp_path / "repeat"
    resumed = tmp_path / "resumed"
    expected = train(full)
    repeated = train(repeat)
    real_epoch = entry.train_epoch

    def interrupt(model, batches, optimizer, args, epoch):
        if epoch == 1:
            raise InterruptedError("test interruption after saved epoch 0")
        return real_epoch(model, batches, optimizer, args, epoch)

    monkeypatch.setattr(entry, "train_epoch", interrupt)
    with pytest.raises(InterruptedError, match="saved epoch 0"):
        train(resumed)
    assert torch.load(resumed / "last.pt", weights_only=True)["epoch"] == 0
    monkeypatch.setattr(entry, "train_epoch", real_epoch)
    continued = train(resumed, "--resume", str(resumed / "last.pt"))

    for checkpoint_name in ("last.pt", "best.pt"):
        reference = torch.load(full / checkpoint_name, weights_only=True)
        for directory in (repeat, resumed):
            actual = torch.load(directory / checkpoint_name, weights_only=True)
            # Runtime output/resume paths intentionally differ; all computational state must match.
            for key in ("model", "optimizer", "rng", "epoch", "best_score", "split_hash"):
                assert_same(reference[key], actual[key])
    assert torch.load(full / "last.pt", weights_only=True)["epoch"] == 1
    for directory in (repeat, resumed):
        assert (full / "epochs.jsonl").read_text() == (directory / "epochs.jsonl").read_text()

    evaluated = inference.main(inference.parser().parse_args([
        "--eval-checkpoint", str(resumed / "best.pt"), "--output", str(tmp_path / "inference"), *runtime,
    ]))
    for result in (repeated, continued, evaluated):
        for key in ("top1_acc", "test_loss", "tokens_per_block"):
            assert_same(expected[key], result[key])
    assert (expected["token_pruning_ratio"] > 0) == (name in {"A4", "B1", "B2"})


def test_a3_paired_inference_repeats_with_actual_a2_checkpoint(tmp_path):
    from A2 import train
    from A3.eval_pruning import parser, run, sha256_file
    from utils.checkpoint import save_checkpoint
    from utils.reproducibility import seed_all

    torch.set_num_threads(2)
    seed_all(42)
    args = train.parser().parse_args(["--output", str(tmp_path / "upstream"), "--device", "cpu",
                                      "--workers", "0", "--weight-bits", "2", "--activation-bits", "8"])
    model = train.build_model(args)
    data = synthetic_data(224)
    with torch.no_grad():
        model(data(args)[0].tensors[0])
    checkpoint = tmp_path / "a2.pt"
    save_checkpoint(checkpoint, model, train.create_optimizer(model, args), 0, 0., args,
                    "reproducibility-fixture")
    original_hash = sha256_file(checkpoint)
    comparisons = []
    for folder in ("first", "repeat"):
        options = parser().parse_args([
            "--checkpoint", str(checkpoint), "--output", str(tmp_path / folder),
            "--device", "cpu", "--workers", "0", "--threads", "2", "--eval-batch-size", "2",
            "--latency-warmup", "0", "--latency-repeats", "1",
        ])
        comparisons.append(run(options, data_factory=data))
    for condition in ("dense", "pruned"):
        first = json.loads((tmp_path / "first" / f"{condition}_metrics.json").read_text())
        repeat = json.loads((tmp_path / "repeat" / f"{condition}_metrics.json").read_text())
        # Timing is a measurement, not a deterministic model output.
        for key in first:
            if not key.startswith("latency_ms"):
                assert_same(first[key], repeat[key])
        if condition == "pruned":
            assert first["tokens_per_block"] == [197, 180, 164, 148, 132, 116, 100, 84, 68, 52, 36, 197]
    for comparison in comparisons:
        assert comparison["backbone_state_unchanged"]
    assert comparisons[0]["top1_loss_pp"] == comparisons[1]["top1_loss_pp"]
    assert sha256_file(checkpoint) == original_hash
