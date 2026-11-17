"""Every QAT experiment evaluates with the quantizers it trained: one grid for both paths.

A2/A4/B2 use affine (asymmetric) fake quantization during training and inference; B1 uses its
LSQ/LSQ+ quantizers in both. A3 restores A2 checkpoints into the same affine quantizers
(tests/test_local_model_policy.py checks exact logits).
"""
import importlib

import pytest
import torch


@pytest.mark.parametrize("name", ["A2", "A4", "B1", "B2"])
def test_training_and_inference_forward_use_the_same_quantization(name, tmp_path):
    torch.set_num_threads(2)
    config = importlib.import_module(f"{name}.config")
    args = config.parser().parse_args(["--output", str(tmp_path), "--image-size", "64", "--drop-path", "0"])
    args.cropr_pruning_rate = args.cls_pruning_rate = 1
    model = importlib.import_module(f"{name}.model").build_model(args)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        model.train()(images)  # Calibrate observers (affine QAT) or initialize LSQ ranges (B1).
        if name != "B1":
            importlib.import_module(f"{name}.quantization").freeze_observers(model, True)
        trained = model.train()(images)
        evaluated = model.eval()(images)
    trained = trained if isinstance(trained, torch.Tensor) else trained[0]
    assert torch.equal(trained, evaluated)
