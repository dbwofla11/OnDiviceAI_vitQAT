"""Linear QAT conversion of the original Cropr ViT: scope, equivalence, gradients, init, metrics."""
import copy
import warnings

import pytest
import timm
import torch
from timm.models.vision_transformer import Attention
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import B1.model as linear_qat_model
from A1 import config as a1_config, model as a1_model
from B1.config import parser
from B1.cropr_adapter import CrossAttention, CroprViT
from B1.metrics import measure_model
from B1.model import build_model, restore_model
from B1.integer import BACKEND, integer_kernel_error
from B1.modules import (QuantConv2d, QuantCrossAttention, QuantLinear, collapsed_quantizer_channels,
                        prepare_linear_qat, project_quantizer_steps, quantizers)
from B1.quantizers import ActivationQuantizer, LSQWeightQuantizer, StatsQWeightQuantizer
from B1.train import train_epoch
from B2 import config as b2_config, metrics as b2_metrics, model as b2_model
from utils.reproducibility import seed_all

BLOCK_LAYERS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def settings(tmp_path, *extra):
    return parser().parse_args([
        "--image-size", "64", "--cropr-pruning-rate", "1", "--device", "cpu", "--epochs", "2",
        "--warmup-epochs", "0", "--batch-size", "2", "--workers", "0", "--drop-path", "0",
        "--latency-warmup", "0", "--latency-repeats", "1", "--output", str(tmp_path / "run"), *extra])


def reference_args(tmp_path, *extra):
    return b2_config.parser().parse_args([
        "--image-size", "64", "--cropr-pruning-rate", "1", "--device", "cpu", "--drop-path", "0",
        "--latency-warmup", "0", "--latency-repeats", "1", "--output", str(tmp_path / "fp32"), *extra])


def fp32_reference(tmp_path, use_cropr=True, llf=False):
    """Same-seed FP32 model of the comparison experiments: B2's Cropr backbone before QAT, or A1."""
    if use_cropr:
        return b2_model.build_backbone(reference_args(tmp_path, *(["--cropr-llf"] if llf else [])))
    return a1_model.build_model(a1_config.parser().parse_args([
        "--image-size", "64", "--drop-path", "0", "--output", str(tmp_path / "fp32")]))


def disable_quantizers(model):
    for quantizer in quantizers(model):
        quantizer.forward = lambda x: x


def batch(n=2):
    return torch.randn(n, 3, 64, 64), torch.arange(n)


def test_quantization_scope_and_bits(tmp_path):
    model = build_model(settings(tmp_path))
    assert type(model) is CroprViT and model.forward.__func__ is CroprViT.forward
    expected = {"patch_embed.proj": 8, "head": 8}
    expected.update({f"blocks.{i}.{layer}": 2 for i in range(12) for layer in BLOCK_LAYERS})
    expected.update({f"cropr.{i}.cross_attn": 8 for i in range(11)})
    found = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantConv2d)):
            assert module.weight_quant.bits == module.input_quant.bits
            found[name] = module.weight_quant.bits
        elif isinstance(module, QuantCrossAttention):
            assert module.query_quant.bits == module.input_quant.bits
            found[name] = module.query_quant.bits
    assert found == expected
    assert isinstance(model.patch_embed.proj, QuantConv2d)
    for block in model.blocks:  # Attention matmuls stay FP32 in Linear QAT.
        assert type(block.attn) is Attention and not block.attn.fused_attn
    for pruner in model.cropr:  # Training-only branches stay FP32.
        assert type(pruner.cross_attn.mlp.fc1) is nn.Linear and type(pruner.head.head) is nn.Linear
    block_quantizers = [model.blocks[0].mlp.fc2.weight_quant, model.blocks[0].mlp.fc2.input_quant]
    assert isinstance(block_quantizers[0], LSQWeightQuantizer) and block_quantizers[0].init == "lsq+"
    assert isinstance(block_quantizers[1], ActivationQuantizer) and block_quantizers[1].asymmetric


def test_full_precision_groups_and_statsq(tmp_path):
    args = settings(tmp_path, "--first-last-bits", "32", "--cropr-scorer-bits", "32",
                    "--weight-quantizer", "statsq", "--activation-quantizer", "lsq",
                    "--weight-bits", "4", "--activation-bits", "8")
    model = build_model(args)
    assert type(model.head) is nn.Linear and type(model.patch_embed.proj) is nn.Conv2d
    assert all(type(p.cross_attn) is CrossAttention for p in model.cropr)
    qkv = model.blocks[3].attn.qkv
    assert isinstance(qkv.weight_quant, StatsQWeightQuantizer) and qkv.weight_quant.bits == 4
    assert not qkv.input_quant.asymmetric and qkv.input_quant.bits == 8
    model = build_model(settings(tmp_path, "--weight-quantizer", "statsq"))
    assert isinstance(model.head.weight_quant, LSQWeightQuantizer)  # 8-bit edges use LSQ+ steps
    assert isinstance(model.cropr[0].cross_attn.query_quant, LSQWeightQuantizer)


@pytest.mark.parametrize("llf", [False, True])
def test_same_initialization_and_disabled_quantizers_match_original(tmp_path, llf):
    seed_all(3)
    reference = fp32_reference(tmp_path, llf=llf)
    seed_all(3)
    model = build_model(settings(tmp_path, *(["--cropr-llf"] if llf else [])))
    reference_parameters = dict(reference.named_parameters())
    quantizer_ids = {id(p) for q in quantizers(model) for p in q.parameters()}
    for name, value in model.named_parameters():
        if id(value) in quantizer_ids:
            continue
        assert torch.equal(value, reference_parameters.pop(name)), name
    assert not reference_parameters
    disable_quantizers(model)
    images, _ = batch()
    for train_outputs in zip(reference.train()(images), model.train()(images)):
        assert torch.allclose(*train_outputs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(reference.eval()(images), model.eval()(images), atol=1e-6, rtol=1e-5)


def test_backbone_initialization_matches_a1(tmp_path):
    seed_all(9)
    dense = fp32_reference(tmp_path, use_cropr=False)
    seed_all(9)
    parameters = dict(build_model(settings(tmp_path)).named_parameters())
    for name, value in dense.named_parameters():
        assert torch.equal(value, parameters[name]), name


def test_scorer_training_and_inference_paths_share_quantized_scores(tmp_path):
    model = build_model(settings(tmp_path))
    attention = model.cropr[0].cross_attn
    tokens = torch.randn(2, 17, 192)
    _, train_scores = attention.train()(tokens)
    assert attention.input_quant.initialized
    assert torch.equal(train_scores, attention.eval().forward_scorer(tokens))
    q_hat = attention.query_quant(attention.queries[0]).detach()
    assert q_hat.unique().numel() <= 2**8


@pytest.mark.parametrize("activation_quantizer", ["lsq+", "lsq"])
def test_training_and_inference_forward_use_the_same_quantization(tmp_path, activation_quantizer):
    model = build_model(settings(tmp_path, "--activation-quantizer", activation_quantizer))
    images, _ = batch()
    with torch.no_grad():
        model.train()(images)  # The first training batch initializes the activation ranges.
        train_logits = model.train()(images)[0]
        eval_logits = model.eval()(images)
    assert torch.equal(train_logits, eval_logits)
    activation = [q for q in quantizers(model) if isinstance(q, ActivationQuantizer)]
    assert all(q.asymmetric == (activation_quantizer == "lsq+") for q in activation)


@pytest.mark.parametrize("weight_bits,activation_bits", [(2, 2), (4, 8)])
def test_training_step_updates_weights_steps_offsets_and_queries(tmp_path, weight_bits, activation_bits):
    args = settings(tmp_path, "--weight-bits", str(weight_bits), "--activation-bits", str(activation_bits))
    model = build_model(args)
    images, labels = batch()
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    # No weight decay: every change below comes from a gradient.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    stats = train_epoch(model, loader, optimizer, args, 0, 1)
    assert stats["train_aux_loss"] > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    changed = {name for name, value in model.named_parameters() if not torch.equal(value, before[name])}
    expected = ["blocks.5.mlp.fc1.weight", "blocks.5.mlp.fc1.weight_quant.step",
                "blocks.5.mlp.fc1.input_quant.step", "cropr.0.cross_attn.queries",
                "cropr.0.cross_attn.query_quant.step", "cropr.0.cross_attn.input_quant.step",
                "head.weight_quant.step"]
    if activation_bits == 2:  # The offset gradient comes from clipped values (LSQ+ Eq. 4).
        expected.append("blocks.5.mlp.fc1.input_quant.offset")
    for name in expected:
        assert name in changed, name
    fc1 = model.blocks[5].mlp.fc1
    assert all(row.unique().numel() <= 2**weight_bits for row in fc1.weight_quant(fc1.weight))
    assert all(q.initialized for q in quantizers(model) if isinstance(q, ActivationQuantizer))


def test_init_checkpoint_loads_fp32_weights_before_step_init(tmp_path):
    seed_all(1)
    reference = fp32_reference(tmp_path)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.mul_(1.5)
    path = tmp_path / "fp32.pt"
    torch.save({"model": reference.state_dict()}, path)
    model = build_model(settings(tmp_path, "--init-checkpoint", str(path)))
    loaded = model.state_dict()
    assert all(torch.equal(value, loaded[name]) for name, value in reference.state_dict().items())
    fc2 = model.blocks[2].mlp.fc2
    assert torch.allclose(fc2.weight_quant.step, LSQWeightQuantizer(fc2.weight, 2).step)
    # An unpruned FP32 backbone (A1) may initialize a Cropr model; Cropr modules keep their init.
    torch.save(fp32_reference(tmp_path, use_cropr=False).state_dict(), path)
    build_model(settings(tmp_path, "--init-checkpoint", str(path)))
    # A checkpoint that already contains quantizer state is not an FP32 initialization.
    torch.save(build_model(settings(tmp_path)).state_dict(), path)
    with pytest.raises(ValueError, match="FP32 model"):
        build_model(settings(tmp_path, "--init-checkpoint", str(path)))
    # Resume and evaluation restore the trained state without reading the initialization file.
    args = settings(tmp_path, "--init-checkpoint", str(tmp_path / "removed.pt"))
    args.eval_checkpoint = "best.pt"
    build_model(args)


def test_pretrained_option_loads_backbone_and_keeps_head_init(tmp_path, monkeypatch):
    create = timm.create_model

    def fake_create(name, pretrained=False, **kwargs):
        model = create(name, pretrained=False, **kwargs)
        if pretrained:  # Stand-in for downloaded weights; timm keeps the new 101-class head.
            with torch.no_grad():
                for key, parameter in model.named_parameters():
                    if not key.startswith("head."):
                        parameter.fill_(0.25)
        return model

    monkeypatch.setattr(linear_qat_model, "create_model", fake_create)
    monkeypatch.setattr(b2_model, "create_model", fake_create)
    seed_all(4)
    scratch = build_model(settings(tmp_path, "--first-last-bits", "32"))
    seed_all(4)
    model = build_model(settings(tmp_path, "--pretrained", "--first-last-bits", "32"))
    assert torch.all(model.blocks[0].attn.qkv.weight == 0.25) and torch.all(model.pos_embed == 0.25)
    assert torch.equal(model.head.weight, scratch.head.weight)
    # B2 builds the same pretrained FP32 initialization for the comparison.
    seed_all(4)
    reference = b2_model.build_backbone(reference_args(tmp_path, "--pretrained"))
    for name, value in reference.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name]), name


def test_restore_rejects_uncalibrated_corrupt_or_foreign_state(tmp_path):
    args = settings(tmp_path)
    model = build_model(args)
    with pytest.raises(ValueError, match="already contains"):
        prepare_linear_qat(model, 2, 2)
    with pytest.raises(ValueError, match="initialized"):
        restore_model(build_model(args), dict(config=vars(args), model=model.state_dict()))
    model.train()(batch()[0])
    state = copy.deepcopy(model.state_dict())
    restored = build_model(args)
    restore_model(restored, dict(config=vars(args), model=state))
    images, _ = batch()
    assert torch.equal(model.eval()(images), restored.eval()(images))
    state["blocks.0.mlp.fc1.weight_quant.step"][0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        restore_model(build_model(args), dict(config=vars(args), model=state))
    with pytest.raises(ValueError, match="linear_qat"):
        restore_model(build_model(args), dict(config=dict(vars(args), mode="qat"), model=state))


def test_metrics_exclude_quantizer_parameters_and_estimate_storage(tmp_path):
    reference = fp32_reference(tmp_path)
    reference_metrics = b2_metrics.measure_model(reference, reference_args(tmp_path))
    sizes = {}
    for bits in (2, 4):
        args = settings(tmp_path, "--weight-bits", str(bits))
        model = build_model(args)
        model.train()(batch()[0])  # initialize activation quantizers
        metrics = measure_model(model, args)
        assert metrics["params"] == reference_metrics["params"]
        assert metrics["inference_params"] == reference_metrics["inference_params"]
        assert metrics["gflops"] == reference_metrics["gflops"]
        assert metrics["tokens_per_block"] == [17] + list(range(15, 4, -1))
        assert metrics["token_pruning_ratio"] == 12 / 16
        assert metrics["latency_backend"] == "fake_quant_fp32"
        if integer_kernel_error() is None:  # Integer kernels depend on the CPU; a reason is reported otherwise.
            assert metrics["integer_latency_ms"] > 0 and metrics["integer_latency_backend"] == BACKEND
        else:
            assert metrics["integer_latency_ms"] is None and metrics["integer_latency_note"]
        assert metrics["collapsed_quantizer_channels"] == 0
        assert metrics["quantizer_params"] == sum(p.numel() for q in quantizers(model) for p in q.parameters())
        assert len(metrics["quantized_layers"]) == 2 + 12 * 4 + 11
        assert metrics["storage_estimated_packed_bytes"] < metrics["params"] * 4
        # Dropping the FP32 training-only branches removes exactly their FP32 bytes.
        auxiliary = metrics["params"] - metrics["inference_params"]
        assert (metrics["storage_estimated_packed_bytes"] - metrics["inference_storage_estimated_packed_bytes"]
                == 4 * auxiliary)
        sizes[bits] = metrics["storage_estimated_packed_bytes"]
    block_weights = sum(block.get_submodule(layer).weight.numel() for block in model.blocks for layer in BLOCK_LAYERS)
    assert sizes[4] - sizes[2] == block_weights * 2 // 8


def test_train_epoch_repairs_and_prevents_collapsed_steps(tmp_path):
    args = settings(tmp_path)
    model = build_model(args)
    images, labels = batch()
    model.train()(images)  # Initialize the activation ranges.
    with torch.no_grad():  # What an AdamW update can do to an 8-bit step (about 4e-4, close to the lr).
        model.head.weight_quant.step[:2] = torch.tensor([-0.02, -0.5])
        model.patch_embed.proj.weight_quant.step[3] = 0.0
        model.cropr[2].cross_attn.query_quant.step[0] = -1e-4
        model.blocks[1].attn.proj.input_quant.step.fill_(-1e-3)  # The whole activation tensor.
    assert collapsed_quantizer_channels(model) == 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    train_epoch(model, DataLoader(TensorDataset(images, labels), batch_size=2), optimizer, args, 0, 1)
    assert collapsed_quantizer_channels(model) == 0
    assert all((q.step > 0).all() for q in quantizers(model) if getattr(q, "step", None) is not None)
    assert torch.isfinite(model.eval()(images)).all()
    for quantizer in quantizers(model):  # The projection never rejects a healthy step again.
        if isinstance(quantizer, LSQWeightQuantizer):
            assert torch.isfinite(quantizer.step).all()


def test_projection_leaves_healthy_steps_bit_identical(tmp_path):
    model = build_model(settings(tmp_path))
    before = {name: value.detach().clone() for name, value in model.named_parameters() if name.endswith(".step")}
    project_quantizer_steps(model)
    assert before and all(torch.equal(value, before[name]) for name, value in model.named_parameters() if name in before)


def test_restoring_a_collapsed_checkpoint_warns_and_is_counted_in_metrics(tmp_path):
    args = settings(tmp_path)
    model = build_model(args)
    model.train()(batch()[0])
    state = copy.deepcopy(model.state_dict())
    state["head.weight_quant.step"][5] = -0.01  # As saved by code without the projection.
    state["blocks.3.mlp.fc1.input_quant.step"].fill_(1e-9)
    restored = build_model(args)
    with pytest.warns(UserWarning, match="2 quantizer channels have a collapsed step"):
        restore_model(restored, dict(config=vars(args), model=state))
    assert measure_model(restored, args)["collapsed_quantizer_channels"] == 2
    # Checkpoints saved before the initial_step buffer existed still restore strictly.
    legacy = {key: value for key, value in state.items() if not key.endswith("initial_step")}
    with pytest.warns(UserWarning, match="collapsed"):
        restore_model(build_model(args), dict(config=vars(args), model=legacy))
    with warnings.catch_warnings():  # A healthy checkpoint restores without any warning.
        warnings.simplefilter("error")
        restore_model(build_model(args), dict(config=vars(args), model=model.state_dict()))
