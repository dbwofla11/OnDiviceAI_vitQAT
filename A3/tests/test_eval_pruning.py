import json

import pytest
import torch

from vitqat.data import make_data
from A3.eval_pruning import (inherited_args, parser, run,
                                 sha256_file, validate_qat_state)
from A3.inference_pruning import PruningConfig
from vitqat.training import save_checkpoint
from A3.tests.test_inference_pruning import fixture_queries, settings, small_fixture_model


@pytest.fixture
def inputs(food_fixture):
    torch.set_num_threads(2)
    args = settings(food_fixture)
    args.data_root = str(food_fixture)
    args.experiment = "upstream-owned-name"
    # Deliberately non-default inherited settings must survive evaluation.
    args.lr, args.epochs, args.label_smoothing = 0.000123, 50, 0.2
    args.weight_bits, args.activation_bits = 2, 8
    model = small_fixture_model(args)
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))
    _, _, _, manifest = make_data(args)
    checkpoint = food_fixture / "best.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), 17, 42., args, manifest["sha256"])
    options = parser().parse_args(["--checkpoint", str(checkpoint),
                                  "--output", str(food_fixture / "pair"), "--device", "cpu",
                                  "--workers", "0", "--threads", "2", "--eval-batch-size", "32",
                                  "--latency-warmup", "0", "--latency-repeats", "1"])
    bundle = None
    return args, model, manifest, bundle, options


@pytest.mark.parametrize("trained_pruning", [False, True])
def test_paired_pipeline_inherits_config_and_preserves_inputs(inputs, trained_pruning):
    args, model, manifest, bundle, options = inputs
    rate = 8 if trained_pruning else 16
    if trained_pruning:
        from A3.pruning import attach_pruning
        checkpoint = torch.load(options.checkpoint, weights_only=True)
        checkpoint["config"].update(cls_pruning="train_and_inference", cls_pruning_rate=rate)
        checkpoint["model"] = attach_pruning(model, pruning_rate=rate, apply_during="train_and_inference").state_dict()
        torch.save(checkpoint, options.checkpoint)
    before = sha256_file(options.checkpoint)
    result = run(options, model_factory=small_fixture_model)
    assert result["backbone_state_unchanged"]
    from pathlib import Path
    output = Path(options.output)
    config = json.loads((output / "evaluation_config.json").read_text())
    assert config["training_config"]["lr"] == args.lr
    assert config["effective_config"]["epochs"] == 50
    assert config["effective_config"]["label_smoothing"] == 0.2
    assert config["effective_config"]["experiment"] == "upstream-owned-name"
    assert config["effective_config"]["weight_bits"] == 2
    assert config["effective_config"]["batch_size"] == 32
    dense = json.loads((output / "dense_metrics.json").read_text())
    pruned = json.loads((output / "pruned_metrics.json").read_text())
    assert dense["checkpoint_sha256"] == pruned["checkpoint_sha256"] == before
    assert dense["split_hash"] == pruned["split_hash"] == manifest["sha256"]
    assert dense["test_samples"] == pruned["test_samples"] == 101
    assert dense["token_pruning_ratio"] == 0
    assert pruned["tokens_per_block"] == [197] + [196 - rate * i for i in range(1, 11)] + [197]
    assert pruned["params"] - dense["params"] == 0
    assert result["top1_loss_pp"] == dense["top1_acc"] - pruned["top1_acc"]
    assert (output / "comparison.csv").exists()
    assert before == sha256_file(options.checkpoint)
    with pytest.raises(ValueError, match="empty"):
        run(options, model_factory=small_fixture_model)


def test_checkpoint_contract_and_observer_validation(inputs):
    args, model, manifest, bundle, options = inputs
    checkpoint = torch.load(options.checkpoint, weights_only=True)
    checkpoint["config"]["use_cropr"] = True
    with pytest.raises(ValueError, match="QAT-only"):
        inherited_args(checkpoint, options)
    uncalibrated = small_fixture_model(args)
    with pytest.raises(ValueError, match="observers"):
        validate_qat_state(uncalibrated)
    from vitqat.quantization import Quantizer
    next(m for m in model.modules() if isinstance(m, Quantizer)).fake_quant.disable_fake_quant()
    with pytest.raises(ValueError, match="fake quantization"):
        validate_qat_state(model)


def test_split_mismatch_fails_before_outputs(inputs):
    from pathlib import Path
    _, _, _, _, options = inputs
    checkpoint = torch.load(options.checkpoint, weights_only=True)
    checkpoint["split_hash"] = "different"
    torch.save(checkpoint, options.checkpoint)
    with pytest.raises(ValueError, match="split"):
        run(options, model_factory=small_fixture_model)
    assert not Path(options.output).exists()
