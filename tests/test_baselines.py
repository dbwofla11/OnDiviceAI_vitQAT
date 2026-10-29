import copy
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vitqat.data import make_data, stratified_split
from vitqat.metrics import measure_dense_vit as measure
from vitqat.models import build_model
from vitqat.train import main, parser
from vitqat.quantization import Quantizer, QATLinear, freeze_observers
from vitqat.training import (evaluate, learning_rate, read_checkpoint,
                               save_checkpoint, seed_all, train_epoch)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def args_for(tmp_path, mode="fp32", **overrides):
    args = parser().parse_args(["--mode", mode, "--output", str(tmp_path / mode)])
    args.device = "cpu"
    args.image_size = 32
    args.epochs = 2
    args.warmup_epochs = 0
    args.workers = 0
    args.batch_size = 64
    args.drop_path = 0.0
    args.latency_repeats = 2
    args.latency_warmup = 0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("weight", [False, True])
def test_quantization_levels_gradient_and_zero(bits, weight):
    q = Quantizer(bits, weight)
    x = torch.linspace(-1, 1, 2048).reshape(2, -1).requires_grad_()
    y = q(x)
    for row in y:
        assert row.unique().numel() <= 2**bits
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(Quantizer(bits, weight)(torch.zeros(2, 8))).all()
    fq = q.fake_quant
    assert fq.quant_min == 0 and fq.quant_max == 2**bits - 1


def test_observers_eval_freeze_and_roundtrip():
    q = Quantizer(2)
    with pytest.raises(RuntimeError, match="observer"):
        q.eval()(torch.ones(8))
    q.train()(torch.linspace(-1, 1, 32))
    previous = copy.deepcopy(q.state_dict())
    q.eval()(torch.ones(32) * 100)
    assert q.fake_quant.observer_enabled.item() == 0
    assert all(torch.equal(value, q.state_dict()[key]) for key, value in previous.items()
               if key != "fake_quant.observer_enabled")
    other = Quantizer(2)
    other.load_state_dict(q.state_dict())
    x = torch.randn(20)
    assert torch.equal(q(x), other.eval()(x))
    freeze_observers(q, True)
    q.train()(torch.ones(32) * 100)
    assert torch.equal(previous["fake_quant.scale"], q.fake_quant.scale)


def test_model_same_initialization_and_disabled_qat_equivalence(tmp_path):
    seed_all(10)
    fp = build_model(args_for(tmp_path))
    seed_all(10)
    qat = build_model(args_for(tmp_path, "qat"))
    assert set(dict(fp.named_parameters())) == set(dict(qat.named_parameters()))
    for name, value in fp.named_parameters():
        assert torch.equal(value, dict(qat.named_parameters())[name])
    x = torch.randn(2, 3, 32, 32)
    qat.train()(x)
    for module in qat.modules():
        if isinstance(module, Quantizer):
            module.fake_quant.disable_fake_quant()
    assert torch.allclose(fp.eval()(x), qat.eval()(x), atol=1e-6, rtol=1e-5)
    assert not any("cropr" in name for name, _ in qat.named_modules())


@pytest.mark.parametrize("weight_bits,activation_bits", [(2, 2), (2, 8), (4, 4), (4, 8)])
def test_model_backward_and_metrics(tmp_path, weight_bits, activation_bits):
    args = args_for(tmp_path, "qat", weight_bits=weight_bits, activation_bits=activation_bits)
    model = build_model(args)
    x = torch.randn(2, 3, 32, 32)
    loss = nn.functional.cross_entropy(model(x), torch.tensor([0, 1]))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    metrics = measure(model, 32, "cpu", warmup=0, repeats=2)
    assert metrics["params"] == sum(p.numel() for p in model.parameters())
    assert metrics["gflops"] > 0 and metrics["latency_ms_median"] > 0
    assert metrics["tokens_per_block"] == [5] * 12
    assert metrics["token_pruning_ratio"] == 0
    assert metrics["latency_backend"] == "fake_quant_fp32"
    assert metrics["storage_estimated_packed_bytes"] < metrics["params"] * 4
    assert metrics["integer_latency_ms"] is None


def test_split_is_stratified_reproducible_disjoint():
    labels = [i for i in range(101) for _ in range(750)]
    train, val = stratified_split(labels, 0.1, 42)
    assert (len(train), len(val)) == (68175, 7575)
    assert not set(train).intersection(val)
    assert len(set(train + val)) == len(labels)
    assert (train, val) == stratified_split(labels, 0.1, 42)
    assert val != stratified_split(labels, 0.1, 43)[1]
    assert all(sum(labels[i] == c for i in val) == 75 for c in range(101))


def test_partial_accumulation_matches_one_batch(tmp_path):
    seed_all(1)
    x, y = torch.randn(5, 4), torch.tensor([0, 1, 0, 1, 0])
    data = TensorDataset(x, y)
    model = nn.Linear(4, 2)
    other = copy.deepcopy(model)
    a = args_for(tmp_path, accum_steps=4, label_smoothing=0.0, clip_grad=100)
    first = train_epoch(model, DataLoader(data, batch_size=2), torch.optim.SGD(model.parameters(), lr=a.lr), a, 0, 1)
    a.accum_steps = 1
    second = train_epoch(other, DataLoader(data, batch_size=5), torch.optim.SGD(other.parameters(), lr=a.lr), a, 0, 1)
    assert first["optimizer_steps"] == second["optimizer_steps"] == 1
    for p, q in zip(model.parameters(), other.parameters()):
        assert torch.allclose(p, q, atol=1e-7)


@pytest.mark.parametrize("mode", ["fp32", "qat"])
def test_checkpoint_resumes_identical_training(tmp_path, mode):
    args = args_for(tmp_path, mode, batch_size=2, accum_steps=2)
    seed_all(5)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data = TensorDataset(torch.randn(5, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4]))
    batches = DataLoader(data, batch_size=2)
    train_epoch(model, batches, optimizer, args, 0, 2)
    path = tmp_path / "checkpoint.pt"
    args.retired_inference_option = True
    save_checkpoint(path, model, optimizer, 0, 12.0, args, "split")
    del args.retired_inference_option
    train_epoch(model, batches, optimizer, args, 1, 2)
    expected = copy.deepcopy(model.state_dict())
    other = build_model(args)
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=args.lr)
    checkpoint = read_checkpoint(path, other, args, "split", other_optimizer)
    assert checkpoint["epoch"] == 0 and checkpoint["best_acc"] == 12.0
    train_epoch(other, batches, other_optimizer, args, 1, 2)
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in expected.items())
    with pytest.raises(ValueError, match="split"):
        read_checkpoint(path, other, args, "wrong")
    args.activation_bits = 8
    with pytest.raises(ValueError, match="activation_bits"):
        read_checkpoint(path, other, args, "split")


@pytest.mark.parametrize("mode", ["fp32", "qat"])
def test_end_to_end_real_food101_loader_on_fixture(food_fixture, mode):
    args = args_for(food_fixture, mode, data_root=str(food_fixture), epochs=1)
    metrics = main(args)
    out = Path(args.output)
    assert metrics["test_samples"] == 101 and math.isfinite(metrics["test_loss"])
    assert 0 <= metrics["top1_acc"] <= 100
    assert metrics["storage_model_state_bytes"] == (out / "model_state.pt").stat().st_size
    assert (out / "metrics.csv").exists() and (out / "split.json").exists()
    assert metrics["weight_bits"] == (32 if mode == "fp32" else 2)
    args.resume = str(out / "last.pt")
    resumed = main(args)
    assert metrics["top1_acc"] == resumed["top1_acc"]
    args.resume = None
    args.eval_checkpoint = str(out / "best.pt")
    args.eval_backend = "pytorch"
    args.output = str(food_fixture / f"{mode}_eval")
    evaluated = main(args)
    assert metrics["top1_acc"] == evaluated["top1_acc"]


def test_fp32_224_flops_and_param_count(tmp_path):
    args = args_for(tmp_path, image_size=224)
    model = build_model(args)
    metrics = measure(model, 224, "cpu", warmup=0, repeats=1)
    # Independent ViT-Tiny formula: patch projection, 12 blocks, two attention matmuls, head.
    n, d = 197, 192
    expected = 2 * 196 * d * (3 * 16 * 16) + 12 * (24 * n * d * d + 4 * n * n * d) + 2 * d * 101
    assert metrics["gflops"] == expected / 1e9
    assert metrics["storage_estimated_packed_bytes"] == metrics["params"] * 4
    assert metrics["latency_backend"] == "fp32"


def test_schedule_boundaries():
    assert learning_rate(0, 10, 0, 1.0, 0.1) == 1.0
    assert learning_rate(9, 10, 2, 1.0, 0.1) == 0.1
    assert learning_rate(0, 10, 2, 1.0, 0.1) == 0.5


def test_qat_224_training_and_eval_no_stat_leakage(tmp_path):
    args = args_for(tmp_path, "qat", image_size=224)
    model = build_model(args)
    output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 101)
    nn.functional.cross_entropy(output, torch.tensor([3])).backward()
    assert model.patch_embed.proj.weight.grad.abs().sum() > 0
    before = {name: value.clone() for name, value in model.named_buffers()
              if name.endswith(("scale", "zero_point", "min_val", "max_val"))}
    evaluate(model, DataLoader(TensorDataset(torch.randn(2, 3, 224, 224) * 20, torch.tensor([1, 2])), batch_size=1), "cpu")
    buffers = dict(model.named_buffers())
    assert all(torch.equal(value, buffers[name]) for name, value in before.items())
    metrics = measure(model, 224, "cpu", warmup=0, repeats=1)
    assert metrics["tokens_per_block"] == [197] * 12
