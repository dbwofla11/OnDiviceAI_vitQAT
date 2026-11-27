"""Changing one experiment's initialization/quantizer requires no shared-code changes."""
import ast
import copy
import importlib
from functools import partial
from pathlib import Path

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import TensorDataset

from utils.runner import run as run_training
from utils.evaluation import run as run_evaluation


def test_timm_pretrained_hub_api_is_available_without_download():
    # timm 1.0.3 catches missing Hub API imports and otherwise reports the
    # installed package as absent. Detect that incompatibility in offline CI too.
    from timm.models._hub import has_hf_hub

    assert has_hf_hub(necessary=True)


@pytest.mark.parametrize("name", ["A1", "A2", "A4", "B1", "B2"])
def test_local_pretrained_factory_preserves_backbone_weights(name, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    config = importlib.import_module(f"{name}.config")
    model_module = importlib.import_module(f"{name}.model")
    args = config.parser().parse_args(["--output", str(tmp_path), "--pretrained"])
    originals, flags = {}, []

    def supplied_model(model_name, *, pretrained, **kwargs):
        flags.append(pretrained)
        base = VisionTransformer(embed_dim=12, depth=12, num_heads=3, **kwargs)
        originals.update({key: value.detach().clone() for key, value in base.named_parameters()})
        return base

    monkeypatch.setattr(model_module, "create_model", supplied_model)
    model = model_module.build_model(args)
    backbone = model.backbone if name == "A4" else model
    for key, value in originals.items():
        assert torch.equal(value, dict(backbone.named_parameters())[key]), key
    assert flags == [True]


def test_pretrained_train_resume_and_inference_do_not_redownload(tmp_path, monkeypatch):
    from A1 import model as other_model
    from A2 import train, inference, model as model_module

    torch.set_num_threads(2)
    other_factory = other_model.create_model
    flags = []

    def supplied_model(model_name, *, pretrained, **kwargs):
        flags.append(pretrained)
        return VisionTransformer(embed_dim=12, depth=12, num_heads=3, **kwargs)

    monkeypatch.setattr(model_module, "create_model", supplied_model)
    data = TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
    data_factory = lambda args: (data, data, data, {"sha256": "pretrained-fixture"})
    monkeypatch.setattr(train, "make_data", data_factory)
    monkeypatch.setattr(inference, "make_data", data_factory)
    runtime = ["--device", "cpu", "--workers", "0", "--batch-size", "2",
               "--latency-warmup", "0", "--latency-repeats", "1"]
    args = train.parser().parse_args(["--output", str(tmp_path / "train"), "--pretrained",
                                     "--image-size", "64", "--epochs", "1", "--warmup-epochs", "0", *runtime])
    trained = train.main(args)
    checkpoint = tmp_path / "train/best.pt"
    saved = copy.deepcopy(torch.load(checkpoint, weights_only=True))
    eval_args = inference.parser().parse_args(["--output", str(tmp_path / "eval"),
                                               "--eval-checkpoint", str(checkpoint), *runtime])
    evaluated = inference.main(eval_args)
    args.resume = str(tmp_path / "train/last.pt")
    resumed = train.main(args)
    assert flags == [True, False, False]
    assert all(result["pretrained"] for result in (trained, evaluated, resumed))
    assert trained["test_loss"] == evaluated["test_loss"] == resumed["test_loss"]
    restored = torch.load(checkpoint, weights_only=True)
    assert all(torch.equal(value, restored["model"][key]) for key, value in saved["model"].items())
    assert other_model.create_model is other_factory


@pytest.mark.parametrize("name", ["A2", "A3", "A4", "B2"])
def test_experiment_owns_its_quantization_and_bit_widths(name):
    module = importlib.import_module(f"{name}.quantization")
    assert module.Quantizer.__module__ == f"{name}.quantization"
    for bits in (2, 3, 4, 8):
        value = torch.linspace(-1, 1, 64, requires_grad=True)
        quantizer = module.Quantizer(bits)
        output = quantizer(value)
        assert output.unique().numel() <= 2 ** bits
        output.sum().backward()
        assert torch.isfinite(value.grad).all()


def test_no_algorithm_dispatch_or_quantization_implementation_in_utils():
    root = Path(__file__).resolve().parents[1]
    # QAT implementations stay experiment-owned. utils/int8.py and utils/tensorrt.py are the shared INT8
    # deployment helpers every experiment imports to time integer kernels; the old module names stay removed.
    for removed in ("models.py", "train.py", "training.py", "quantization.py", "tensorrt_int8.py",
                    "tensorrt_benchmark.py", "pruning_metrics.py"):
        assert not (root / "utils" / removed).exists()
    for path in (root / "utils").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert "quantization" not in (node.module or ""), path
    for experiment in ("A1", "A2", "A3", "A4", "B1", "B2"):
        for path in (root / experiment).rglob("*.py"):
            if "tests" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = (node.module or "").split(".")[0]
                    assert imported not in {"A1", "A2", "A3", "A4", "B1", "B2"} - {experiment}, path


def test_a2_checkpoint_loads_into_a3_owned_quantization(tmp_path, monkeypatch):
    from A2 import config, model as source_module
    from A3 import model as target_module
    from A2.quantization import Quantizer as SourceQuantizer
    from A3.quantization import Quantizer as TargetQuantizer

    torch.set_num_threads(2)
    def factory(name, *, pretrained, **kwargs):
        assert not pretrained
        return VisionTransformer(embed_dim=12, depth=12, num_heads=3, **kwargs)

    monkeypatch.setattr(source_module, "create_model", factory)
    monkeypatch.setattr(target_module, "create_model", factory)
    args = config.parser().parse_args(["--output", str(tmp_path), "--image-size", "64", "--drop-path", "0"])
    source = source_module.build_model(args)
    images = torch.randn(2, 3, 64, 64)
    source(images)
    path = tmp_path / "a2.pt"
    torch.save({"model": source.state_dict(), "config": vars(args)}, path)
    args.eval_checkpoint = str(path)
    target = target_module.build_model(args)
    target.load_state_dict(torch.load(path, weights_only=True)["model"], strict=True)
    assert any(isinstance(module, TargetQuantizer) for module in target.modules())
    assert not any(isinstance(module, SourceQuantizer) for module in target.modules())
    with torch.no_grad():
        torch.testing.assert_close(source.eval()(images), target.eval()(images), rtol=0, atol=0)
