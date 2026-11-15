"""Checkpoint representation and algorithm dependencies are experiment-owned."""
import copy
import importlib
import subprocess
import sys

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer


@pytest.fixture(params=["A2", "A3", "A4", "B2"])
def qat_checkpoint(request, monkeypatch):
    torch.set_num_threads(2)
    name = request.param
    config = importlib.import_module(f"{'A2' if name == 'A3' else name}.config")
    module = importlib.import_module(f"{name}.model")
    quantization = importlib.import_module(f"{name}.quantization")
    args = config.parser().parse_args(["--output", "unused", "--image-size", "64",
                                      "--drop-path", "0", "--activation-bits", "8"])
    args.cropr_pruning_rate = args.cls_pruning_rate = 1
    args.eval_checkpoint = "checkpoint.pt"

    def factory(name, *, pretrained, **kwargs):
        assert not pretrained
        return VisionTransformer(embed_dim=12, depth=3, num_heads=3, **kwargs)

    monkeypatch.setattr(module, "create_model", factory)
    model = module.build_model(args)
    images = torch.randn(2, 3, 64, 64)
    model(images)
    checkpoint = copy.deepcopy(dict(model=model.state_dict(), config=vars(args)))
    return module, quantization, args, model, images, checkpoint


@pytest.mark.parametrize("legacy", [False, True])
def test_restore_preserves_logits_and_legacy_state(qat_checkpoint, legacy):
    module, quantization, args, source, images, checkpoint = qat_checkpoint
    if legacy:
        checkpoint["model"] = {k: v for k, v in checkpoint["model"].items()
                               if not k.endswith("._bit_width")}
    before = copy.deepcopy(checkpoint)
    restored = module.build_model(args)
    module.restore_model(restored, checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(source.eval()(images), restored.eval()(images), rtol=0, atol=0)
    assert checkpoint["model"].keys() == before["model"].keys()
    assert all(torch.equal(v, before["model"][k]) for k, v in checkpoint["model"].items())


def test_changed_bit_config_cannot_reinterpret_saved_state(qat_checkpoint):
    module, _, args, _, _, checkpoint = qat_checkpoint
    checkpoint["config"]["activation_bits"] = args.activation_bits = 4
    restored = module.build_model(args)
    with pytest.raises(RuntimeError, match="Quantization bit width mismatch"):
        module.restore_model(restored, checkpoint)


@pytest.mark.parametrize("key", ["mode", "weight_bits", "activation_bits", "qat_scope"])
def test_missing_quantization_metadata_is_not_guessed(qat_checkpoint, key):
    module, _, _, model, _, checkpoint = qat_checkpoint
    del checkpoint["config"][key]
    with pytest.raises(ValueError, match="requires|Missing quantization"):
        module.restore_model(model, checkpoint)


@pytest.mark.parametrize("corruption", ["disabled", "uncalibrated", "nan_scale", "zero_point"])
def test_invalid_quantized_state_is_rejected_at_load(qat_checkpoint, corruption):
    module, _, _, model, _, checkpoint = qat_checkpoint
    suffix, value = {
        "disabled": ("fake_quant.fake_quant_enabled", 0),
        "uncalibrated": ("fake_quant.activation_post_process.min_val", float("inf")),
        "nan_scale": ("fake_quant.scale", float("nan")),
        "zero_point": ("fake_quant.zero_point", 10000),
    }[corruption]
    key = next(key for key in checkpoint["model"] if key.endswith(suffix))
    checkpoint["model"][key].fill_(value)
    with pytest.raises(ValueError, match="observers|fake quantization"):
        module.restore_model(model, checkpoint)


def test_qat_cannot_be_applied_twice(qat_checkpoint):
    _, quantization, args, model, _, _ = qat_checkpoint
    with pytest.raises(ValueError, match="already contains"):
        quantization.prepare_qat(model, args.weight_bits, args.activation_bits)


@pytest.mark.parametrize("key", ["mode", "use_cropr", "weight_bits", "activation_bits", "qat_scope"])
def test_inference_rejects_missing_model_metadata_before_data(tmp_path, monkeypatch, key):
    from A2 import inference, config
    from functools import partial
    from utils.evaluation import run

    saved = vars(config.parser().parse_args(["--output", "unused"]))
    del saved[key]
    checkpoint = tmp_path / "incomplete.pt"
    torch.save(dict(config=saved), checkpoint)

    def forbidden(args):
        pytest.fail("Invalid checkpoint must fail before loading a dataset")

    monkeypatch.setattr(inference, "make_data", forbidden)
    args = inference.parser().parse_args(["--eval-checkpoint", str(checkpoint),
                                         "--output", str(tmp_path / "eval")])
    with pytest.raises(ValueError, match="Missing model config"):
        inference.main(args)
    assert not (tmp_path / "eval").exists()


@pytest.mark.parametrize("name,scope", [("A2", "all"), ("B2", "all"), ("B2", "backbone")])
def test_local_export_conversion_after_checkpoint_restore(name, scope, monkeypatch):
    torch.set_num_threads(2)
    config = importlib.import_module(f"{name}.config")
    module = importlib.import_module(f"{name}.model")
    quantization = importlib.import_module(f"{name}.quantization")
    exporter = importlib.import_module(f"{name}.tensorrt_int8")
    args = config.parser().parse_args(["--output", "unused", "--image-size", "64", "--activation-bits", "8"])
    args.cropr_pruning_rate, args.qat_scope = 1, scope

    def factory(name, *, pretrained, **kwargs):
        return VisionTransformer(embed_dim=12, depth=3, num_heads=3, **kwargs)

    monkeypatch.setattr(module, "create_model", factory)
    source = module.build_model(args)
    images = torch.randn(2, 3, 64, 64)
    source(images)
    restored = module.build_model(args)
    module.restore_model(restored, dict(config=vars(args), model=source.state_dict()))
    converted = exporter.convert_tensorrt_int8(restored)
    assert not any(isinstance(m, quantization.Quantizer) for m in converted.modules())
    with torch.no_grad():
        assert converted(images).shape == (2, 101)
        assert torch.isfinite(converted(images)).all()


def test_a_series_import_and_model_build_do_not_load_cropr():
    # A fresh process catches transitive dependencies hidden by pytest's imports.
    script = '''
import importlib
import sys
import torch
torch.set_num_threads(1)
for name in ("A1", "A2", "A3", "A4"):
    config = importlib.import_module(f"{'A2' if name == 'A3' else name}.config")
    model = importlib.import_module(f"{name}.model")
    importlib.import_module(f"{name}.inference")
    if name in ("A2", "A3"):
        importlib.import_module(f"{name}.benchmark_tensorrt_int8")
    args = config.parser().parse_args(["--output", "unused"])
    model.build_model(args)
assert not any(name == "cropr" or name.startswith("cropr.") for name in sys.modules)
assert "utils.cropr_adapter" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("name", ["A1", "A2", "A3", "A4"])
def test_native_backbone_restores_old_cropr_adapter_weights(name, monkeypatch):
    from B2.cropr_adapter import adapt_model

    torch.set_num_threads(2)
    config = importlib.import_module(f"{'A2' if name == 'A3' else name}.config")
    module = importlib.import_module(f"{name}.model")
    args = config.parser().parse_args(["--output", "unused", "--image-size", "64", "--drop-path", "0"])
    args.cls_pruning_rate = 1
    base = VisionTransformer(img_size=64, embed_dim=12, depth=3, num_heads=3, num_classes=101)
    old = adapt_model(base, image_size=64, drop_path=0)
    if name != "A1":
        quant = importlib.import_module(f"{name}.quantization")
        quant.prepare_qat(old, args.weight_bits, args.activation_bits)
    if name == "A4":
        pruning = importlib.import_module("A4.pruning")
        old = pruning.attach_pruning(old, config=pruning.PruningConfig(pruning_rate=1),
                                     apply_during="train_and_inference")
    images = torch.randn(2, 3, 64, 64)
    old(images)
    monkeypatch.setattr(module, "create_model", lambda *a, **kw: copy.deepcopy(base))
    new = module.build_model(args)
    module.restore_model(new, dict(config=vars(args), model=old.state_dict()))
    with torch.no_grad():
        torch.testing.assert_close(old.eval()(images), new.eval()(images), rtol=0, atol=0)
