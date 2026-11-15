"""Adding experiment behavior must not require edits to the common runner."""
import ast
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from tests.support import parser
from utils.runner import run
from utils.classification import evaluate
from utils.results import save_results
from utils.data import loader


def test_runner_accepts_independent_model_training_data_and_metrics(tmp_path):
    torch.set_num_threads(2)
    args = parser().parse_args([
        "--mode", "fp32", "--experiment", "new-experiment",
        "--output", str(tmp_path / "new"), "--device", "cpu", "--epochs", "1",
        "--warmup-epochs", "0", "--workers", "0", "--batch-size", "2",
    ])
    calls = []
    # The infrastructure does not require any experiment/method selector.
    del args.mode, args.use_cropr, args.cls_pruning, args.weight_bits, args.activation_bits
    del args.lr, args.min_lr, args.warmup_epochs, args.clip_grad, args.accum_steps
    del args.mixup, args.cutmix, args.label_smoothing, args.weight_decay
    del args.image_size  # Reporting must not require an image-model configuration.

    def make_model(config):
        calls.append("model")
        return nn.Linear(4, 2)  # No timm-specific methods or QAT attributes.

    def make_dataset(config):
        calls.append("data")
        data = TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))
        return data, data, data, {"sha256": "fixture-only"}

    def custom_train(model, batches, optimizer, config, epoch):
        calls.append("train")
        # This recipe intentionally has no augmentation, warmup or gradient clipping.
        model.train()
        for images, target in batches:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), target)
            loss.backward()
            optimizer.step()
        return {"train_loss": loss.item()}

    def custom_optimizer(model, config):
        calls.append("optimizer")
        return torch.optim.SGD(model.parameters(), lr=0.05)

    def custom_eval(*arguments):
        calls.append("evaluate")
        return evaluate(*arguments)

    def custom_metrics(model, config):
        calls.append("metrics")
        return {"token_pruning_ratio": 0.5, "tokens_per_block": [5, 3], "gflops": 0.123,
                "params": sum(p.numel() for p in model.parameters()), "custom_measurement": 42}

    def custom_report(model, config, test, checkpoint, selected, manifest, output, metrics):
        calls.append("report")
        return save_results(dict(metrics, experiment=config.experiment, quality=0.75), output)

    metrics = run(args, build_model=make_model, measure_model=custom_metrics,
                  data_factory=make_dataset, loader_factory=loader, selection_score=lambda val: val["top1_acc"], train_epoch_fn=custom_train, evaluate_fn=custom_eval,
                  optimizer_factory=custom_optimizer, report_results=custom_report)
    assert calls == ["data", "model", "optimizer", "train", "evaluate", "evaluate", "metrics", "report"]
    assert metrics["experiment"] == "new-experiment"
    assert metrics["token_pruning_ratio"] == 0.5 and metrics["tokens_per_block"] == [5, 3]
    assert metrics["gflops"] == 0.123 and metrics["custom_measurement"] == 42
    assert metrics["quality"] == 0.75 and "image_size" not in metrics
    assert not (tmp_path / "new/model_state.pt").exists()


def test_shared_code_does_not_import_experiments():
    root = Path(__file__).resolve().parents[1] / "utils"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in {"A1", "A2", "A3", "A4", "B1", "B2", "vitqat", "experiments"} for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"A1", "A2", "A3", "A4", "B1", "B2", "vitqat", "experiments"}, path


def test_evaluation_accepts_nonclassification_report_and_custom_artifact(tmp_path):
    from types import SimpleNamespace
    from utils.evaluation import run as run_evaluation

    source = nn.Linear(4, 1)
    checkpoint = tmp_path / "source.pt"
    # Neither training-selection metadata nor image-model settings are needed.
    torch.save(dict(config={"seed": 42}, model=source.state_dict(), split_hash="custom"), checkpoint)
    args = SimpleNamespace(eval_checkpoint=str(checkpoint), resume=None, experiment="regression",
                           output=str(tmp_path / "eval"), device="cpu", batch_size=2,
                           workers=0, threads=1, latency_warmup=0, latency_repeats=1)
    data = TensorDataset(torch.zeros(2, 4), torch.zeros(2, 1))

    def evaluate_regression(model, batches, device):
        errors = [(model(x) - y).square().sum().item() for x, y in batches]
        return {"mse": sum(errors) / len(data)}

    def custom_report(model, config, test, saved, selected, manifest, output, metrics):
        (output / "weights.txt").write_text(str(model.weight.tolist()), encoding="utf-8")
        return save_results(dict(metrics, **test, artifact="weights.txt"), output)

    result = run_evaluation(args, build_model=lambda args: nn.Linear(4, 1),
                            evaluate_fn=evaluate_regression,
                            measure_model=lambda model, args: {"parameters": 5},
                            report_results=custom_report, validate_config=lambda args: None, loader_factory=loader,
                            data_factory=lambda args: (data, data, data, {"sha256": "custom"}))
    assert set(result) == {"parameters", "mse", "artifact"}
    assert (tmp_path / "eval/weights.txt").is_file()
    assert not (tmp_path / "eval/model_state.pt").exists()
