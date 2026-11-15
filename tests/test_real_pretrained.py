"""Opt-in integration checks using actual timm pretrained weights, without model mocks.

PowerShell:
    $env:RUN_PRETRAINED_INTEGRATION = "1"
    python -m pytest tests/test_real_pretrained.py -q -s

The first run may download weights to timm/Hugging Face's normal cache.
Synthetic inputs check loading/forward compatibility, not Food-101 accuracy.
"""
import copy
import hashlib
import importlib
import os

import pytest
import timm
import torch


MODEL = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PRETRAINED_INTEGRATION") != "1",
    reason="Set RUN_PRETRAINED_INTEGRATION=1 to load actual pretrained weights",
)


@pytest.fixture(scope="module")
def pretrained_backbone():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    print(f"Loading actual timm pretrained weights: {MODEL}", flush=True)
    reference = timm.create_model(MODEL, pretrained=True)
    weights = {name: value.detach().clone() for name, value in reference.named_parameters()
               if not name.startswith("head.")}
    scratch = timm.create_model(MODEL, pretrained=False)
    assert not torch.equal(weights["patch_embed.proj.weight"], scratch.patch_embed.proj.weight)
    digest = hashlib.sha256()
    for name, value in weights.items():
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    print(f"Pretrained backbone: {len(weights)} tensors, sha256={digest.hexdigest()}", flush=True)
    del reference, scratch
    try:
        yield weights
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("experiment", ["A1", "A2", "A4", "B1", "B2"])
def test_real_pretrained_weights_and_forward(experiment, pretrained_backbone, tmp_path):
    config = importlib.import_module(f"{experiment}.config")
    implementation = importlib.import_module(f"{experiment}.model")
    args = config.parser().parse_args([
        "--output", str(tmp_path), "--pretrained", "--model", MODEL,
        "--device", "cpu", "--drop-path", "0",
    ])
    model = implementation.build_model(args)
    backbone = model.backbone if experiment == "A4" else model
    parameters = dict(backbone.named_parameters())
    for name, expected in pretrained_backbone.items():
        assert torch.equal(parameters[name], expected), f"{experiment}: pretrained weight changed: {name}"
    assert backbone.head.out_features == 101

    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model.train()(images)  # Initialize QAT observers, including Cropr auxiliary branches.
        logits = model.eval()(images)
    assert logits.shape == (1, 101)
    assert torch.isfinite(logits).all()
    print(f"{experiment}: pretrained tensors match exactly; finite forward {tuple(logits.shape)}", flush=True)


def test_a3_restores_actual_pretrained_a2_checkpoint(pretrained_backbone, tmp_path):
    from A2 import config, model as a2
    from A3 import model as a3

    args = config.parser().parse_args([
        "--output", str(tmp_path), "--pretrained", "--model", MODEL,
        "--device", "cpu", "--drop-path", "0", "--activation-bits", "8",
    ])
    source = a2.build_model(args)
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        source(images)
        expected = source.eval()(images)
    path = tmp_path / "pretrained_a2.pt"
    torch.save(dict(config=vars(args), model=source.state_dict()), path)
    restored_args = copy.copy(args)
    restored_args.eval_checkpoint = str(path)
    restored = a3.build_model(restored_args)
    a3.restore_model(restored, torch.load(path, map_location="cpu", weights_only=True))
    parameters = dict(restored.named_parameters())
    for name, value in pretrained_backbone.items():
        assert torch.equal(parameters[name], value), name
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(images), expected, rtol=0, atol=0)
    print("A3: actual pretrained A2 checkpoint restored; logits match exactly", flush=True)
