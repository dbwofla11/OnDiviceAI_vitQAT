"""Exercise infrastructure without the built-in ViT/Food-101 experiment policies."""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def test_infrastructure_does_not_import_model_or_dataset_libraries():
    script = '''
import importlib
import sys
import importlib.abc
class BlockDomainImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"timm", "torchvision", "cropr", "A1", "A2", "A3", "A4", "B1", "B2"}:
            raise AssertionError("Unexpected dependency: " + fullname)
sys.meta_path.insert(0, BlockDomainImports())
for name in ("runner", "evaluation", "checkpoint", "results", "reproducibility", "optim",
             "cli", "metrics", "classification"):
    importlib.import_module("utils." + name)
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
                   cwd=Path(__file__).resolve().parents[1])


def test_no_experiment_or_cropr_imports_in_any_utils_source():
    root = Path(__file__).resolve().parents[1] / "utils"
    assert not (root / "cropr_adapter.py").exists()
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            assert not any(name.split(".")[0] in {"cropr", "A1", "A2", "A3", "A4", "B1", "B2"}
                           for name in imported), path


@pytest.mark.parametrize("count", [2, 3])
def test_mixup_uses_requested_class_count_for_even_and_odd_batches(count):
    from utils.augment import build_mixup, augment_batch

    mixup = build_mixup(0.8, 0, 0.1, num_classes=3)
    images = torch.randn(count, 3, 8, 8)
    labels = torch.arange(count) % 3
    _, soft = augment_batch(mixup, images, labels, 0.1)
    assert soft.shape == (count, 3)
    torch.testing.assert_close(soft.sum(-1), torch.ones(count))


def test_seeding_does_not_change_backend_precision_policy():
    from utils.reproducibility import seed_all, configure_backend

    before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        configure_backend(deterministic=False, allow_tf32=True)
        seed_all(7)
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        configure_backend(deterministic=True, allow_tf32=False)
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = before


def test_regression_training_selects_lowest_error_and_resumes_without_task_defaults(tmp_path):
    from utils.runner import run

    args = SimpleNamespace(eval_checkpoint=None, resume=None, output=str(tmp_path / "run"),
                           device="cpu", batch_size=1, workers=0, epochs=3, threads=1,
                           seed=7, latency_repeats=1, latency_warmup=0)
    data = TensorDataset(torch.ones(1, 1), torch.zeros(1, 1))
    trained_epochs = []

    def train_epoch(model, batches, optimizer, config, epoch):
        with torch.no_grad():
            model.weight.fill_([2.0, 1.0, 3.0][epoch])
        trained_epochs.append(epoch)
        return {"training_mse": model.weight.item() ** 2}

    def report(model, args, test, checkpoint, selected, manifest, output, metrics):
        return dict(metrics, selected_epoch=checkpoint["epoch"], best_score=checkpoint["best_score"])

    kwargs = dict(build_model=lambda args: nn.Linear(1, 1, bias=False),
                  optimizer_factory=lambda model, args: torch.optim.SGD(model.parameters(), lr=0.1),
                  train_epoch_fn=train_epoch,
                  evaluate_fn=lambda model, batches, device: {"mse": model.weight.item() ** 2},
                  selection_score=lambda validation: -validation["mse"],
                  measure_model=lambda model, args: {"weight": model.weight.item()},
                  report_results=report,
                  data_factory=lambda args: (data, data, data, {"sha256": "regression"}),
                  loader_factory=lambda dataset, args, **kw: DataLoader(dataset, batch_size=1))
    result = run(args, **kwargs)
    assert result == {"weight": 1.0, "selected_epoch": 1, "best_score": -1.0}
    assert trained_epochs == [0, 1, 2]
    args.resume = str(tmp_path / "run/last.pt")
    assert run(args, **kwargs) == result
    assert trained_epochs == [0, 1, 2]
    saved = torch.load(tmp_path / "run/best.pt", weights_only=True)
    assert "best_score" in saved and "best_acc" not in saved


def test_legacy_best_accuracy_checkpoint_remains_readable(tmp_path):
    from utils.checkpoint import read_checkpoint

    model = nn.Linear(1, 1)
    path = tmp_path / "legacy.pt"
    torch.save(dict(model=model.state_dict(), config={}, split_hash="legacy", best_acc=73.0), path)
    checkpoint = read_checkpoint(path, model, SimpleNamespace(), "legacy")
    assert checkpoint["best_score"] == 73.0
    assert "best_score" not in torch.load(path, weights_only=True)
