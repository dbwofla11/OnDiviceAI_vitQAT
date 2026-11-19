"""Exercise the two experiment stages and their checkpoint handoff."""
import json
from argparse import Namespace

import pytest
import torch
from torchvision.transforms.functional import to_pil_image
from timm.models.vision_transformer import VisionTransformer

from B3 import inference, model, predict, ptq, train
from B3.quantization import Quantizer
from utils.data import make_food101_data


def test_single_image_preprocessing_matches_evaluation(food_fixture):
    args = train.parser().parse_args(["--data-root", str(food_fixture), "--output", "unused"])
    _, _, test_data, _ = make_food101_data(args)
    generator = torch.Generator().manual_seed(42)
    image = to_pil_image(torch.randint(256, (3, 263, 347), generator=generator, dtype=torch.uint8))
    assert torch.equal(predict.food101_eval_transform(args.image_size)(image), test_data.transform(image))


def test_2a_training_defaults():
    args = train.parser().parse_args(["--output", "unused"])
    assert (args.mode, args.model, args.cropr_pruning_rate, args.cropr_llf) == (
        "fp32", "vit_small_patch16_224", 16, True)
    assert (args.seed, args.epochs, args.mixup, args.cutmix, args.random_erasing) == (
        42, 300, 0.8, 1.0, 0.25)


def test_fp32_train_calibrate_restore(food_fixture, tmp_path, monkeypatch):
    def small_vit(name, pretrained, **kwargs):
        return VisionTransformer(embed_dim=12, depth=4, num_heads=3, **kwargs)

    monkeypatch.setattr(model, "create_model", small_vit)
    runtime = ["--device", "cpu", "--data-root", str(food_fixture), "--workers", "0",
               "--batch-size", "16", "--threads", "2"]
    timing = ["--latency-warmup", "0", "--latency-repeats", "1"]
    run = tmp_path / "fp32"
    fp32 = train.main(train.parser().parse_args([
        "--output", str(run), "--image-size", "64", "--cropr-pruning-rate", "1",
        "--epochs", "1", "--warmup-epochs", "0", *runtime, *timing]))
    path = run / "best.pt"
    original = torch.load(path, weights_only=True)
    assert fp32["latency_backend"] == "fp32"
    assert not any("quant" in name for name in original["model"])
    split = json.loads((run / "split.json").read_text())
    opts = ptq.parser().parse_args(["--checkpoint", str(path), "--output", str(tmp_path / "ptq"),
                                    "--calibration-samples", "17", *runtime])
    converted = ptq.main(opts)
    saved = torch.load(converted, weights_only=True)
    indices = saved["calibration"]["train_indices"]
    assert len(set(indices)) == saved["calibration"]["samples"] == 17
    assert set(indices) <= set(split["train_indices"])
    assert not set(indices).intersection(split["val_indices"])
    for key, value in original["model"].items():
        assert torch.equal(value, saved["model"][key]), key
    restored = model.build_model(Namespace(**saved["config"]))
    model.restore_model(restored, saved)
    scorer = restored.cropr[0].cross_attn
    assert isinstance(scorer.query_quant, Quantizer)
    quantizers = [m for m in restored.modules() if isinstance(m, Quantizer)]
    assert all(q.fake_quant_enabled.item() and not q.observer_enabled.item() for q in quantizers)
    metrics = inference.main(inference.parser().parse_args([
        "--output", str(tmp_path / "eval"), "--eval-checkpoint", str(converted), *runtime, *timing]))
    assert metrics["mode"] == "ptq" and metrics["latency_backend"] == "fake_quant_fp32"
    assert (metrics["test_samples"], metrics["weight_bits"], metrics["activation_bits"]) == (101, 2, 8)
    assert metrics["tokens_per_block"] == fp32["tokens_per_block"] == [17, 15, 14, 17]
    image = food_fixture / "food-101/images/class_000/test_0.jpg"
    for checkpoint in (path, converted):
        predictions = predict.predict(checkpoint, image, "cpu", 3)
        assert len(predictions) == 3 and predictions[0]["class"] in split["classes"]
    opts.checkpoint = str(converted)
    opts.output = str(tmp_path / "twice")
    with pytest.raises(ValueError, match="FP32 source"):
        ptq.main(opts)
