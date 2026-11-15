import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from B2.cropr_adapter import CroprViT
from tests.support import parser
from tests.support import measure_vit
from tests.support import build_model
from B2.quantization import Quantizer
from tests.support import measure_model, train_epoch
from tests.support import run
from utils.classification import evaluate
from tests.support import read_checkpoint
from utils.checkpoint import save_checkpoint
from utils.reproducibility import seed_all


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def settings(tmp_path, mode="qat", llf=False, scope="all"):
    args = parser().parse_args([
        "--mode", mode, "--use-cropr", "--cropr-pruning-rate", "1",
        "--image-size", "64", "--device", "cpu", "--epochs", "2", "--warmup-epochs", "0",
        "--batch-size", "2", "--workers", "0", "--drop-path", "0",
        "--qat-scope", scope, "--output", str(tmp_path / "cropr"),
    ])
    args.cropr_llf = llf
    return args


@pytest.mark.parametrize("mode", ["fp32", "qat"])
@pytest.mark.parametrize("llf", [False, True])
def test_original_forward_auxiliary_training_metrics_and_restore(tmp_path, mode, llf):
    args = settings(tmp_path, mode, llf)
    model = build_model(args)
    assert type(model) is CroprViT
    assert model.forward.__func__ is CroprViT.forward
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    batches = DataLoader(TensorDataset(images, labels), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    before = model.cropr[0].cross_attn.queries.detach().clone()
    stats = train_epoch(model, batches, optimizer, args, 0, 1)
    assert stats["train_aux_loss"] > 0
    assert stats["train_loss"] == pytest.approx(stats["train_main_loss"] + stats["train_aux_loss"])
    assert not torch.equal(before, model.cropr[0].cross_attn.queries)
    result = evaluate(model, batches, "cpu")
    assert result["samples"] == 2 and result["loss"] > 0
    metrics = measure_vit(model, 64, "cpu", 0, 1)
    expected = [17] + list(range(15, 4, -1))
    if llf:
        expected[-1] = 17
    assert metrics["tokens_per_block"] == expected
    assert metrics["token_pruning_ratio"] == (11 if llf else 12) / 16
    assert metrics["final_token_pruning_ratio"] == (0 if llf else 12 / 16)
    assert metrics["inference_params"] < metrics["params"]
    # Independent formula using actual block sequence plus original Cropr scorer costs.
    d = 192
    scorer_inputs = expected[:10 if llf else 11]
    flops = 2 * 16 * d * (3 * 16 * 16) + 2 * d * 101
    flops += sum(24 * n * d * d + 4 * n * n * d for n in expected)
    flops += sum(2 * n * d for n in scorer_inputs)
    assert metrics["gflops"] == flops / 1e9
    reference = model.eval()(images).detach()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0, result["top1_acc"], args, "split")
    restored = build_model(args)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=args.lr)
    read_checkpoint(path, restored, args, "split", restored_optimizer)
    assert torch.equal(reference, restored.eval()(images))
    train_epoch(model, batches, optimizer, args, 1, 1)
    train_epoch(restored, batches, restored_optimizer, args, 1, 1)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("scope", ["all", "backbone"])
def test_scope_and_disabled_qat_matches_original(tmp_path, scope):
    seed_all(7)
    fp = build_model(settings(tmp_path, "fp32"))
    seed_all(7)
    qat = build_model(settings(tmp_path, "qat", scope=scope))
    for name, value in fp.named_parameters():
        assert torch.equal(value, dict(qat.named_parameters())[name])
    pruner_quantizers = [m for m in qat.cropr.modules() if isinstance(m, Quantizer)]
    assert bool(pruner_quantizers) == (scope == "all")
    images = torch.randn(2, 3, 64, 64)
    qat.train()(images)
    for module in qat.modules():
        if isinstance(module, Quantizer):
            module.fake_quant.disable_fake_quant()
    assert torch.allclose(fp.eval()(images), qat.eval()(images), atol=1e-6, rtol=1e-5)


def test_main_loss_alone_does_not_train_pruner_queries(tmp_path):
    model = build_model(settings(tmp_path, "fp32"))
    labels = torch.tensor([0, 1])
    outputs = model(torch.randn(2, 3, 64, 64))
    nn.functional.cross_entropy(outputs[0], labels).backward(retain_graph=True)
    queries = model.cropr[0].cross_attn.queries
    assert queries.grad is None
    sum(nn.functional.cross_entropy(pred, labels) for pred in outputs[1:]).backward()
    assert torch.isfinite(queries.grad).all() and queries.grad.abs().sum() > 0


def test_invalid_schedule_rejected(tmp_path):
    args = settings(tmp_path)
    args.image_size = 32  # Five tokens cannot support the original 12-token removal schedule.
    with pytest.raises(ValueError, match="remove all tokens"):
        build_model(args)


def test_backbone_weights_preserved_when_enabling_cropr(tmp_path):
    args = settings(tmp_path, "fp32")
    args.use_cropr = False
    seed_all(23)
    unpruned = build_model(args)
    args.use_cropr = True
    seed_all(23)
    pruned = build_model(args)
    for name, value in unpruned.named_parameters():
        assert torch.equal(value, dict(pruned.named_parameters())[name])


@pytest.mark.parametrize("enabled", [False, True])
def test_original_constructor_matches_reused_timm_model(tmp_path, enabled):
    args = settings(tmp_path, "fp32")
    args.use_cropr = enabled
    reused = build_model(args)
    cfg = dict(use_cropr=enabled, llf=False, pruning_rate=1, num_queries=1,
               num_heads=1, pre_attn_norm=False, q_proj=False, k_proj=False,
               v_proj=False, mlp=True, mlp_ratio=4, training=True)
    original = CroprViT(cfg, img_size=64, patch_size=16, num_classes=101,
                       embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                       drop_path_rate=0.0, global_pool="token")
    original.load_state_dict(reused.state_dict(), strict=True)
    images = torch.randn(2, 3, 64, 64)
    assert torch.allclose(original.eval()(images), reused.eval()(images), atol=1e-6, rtol=1e-5)


def test_cropr_qat_pipeline(tmp_path):
    args = settings(tmp_path)
    args.epochs, args.latency_warmup, args.latency_repeats = 1, 0, 1
    args.experiment = "cropr_qat"

    def data_factory(config):
        splits = [TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])) for _ in range(3)]
        return *splits, {"sha256": "generated-fixture"}

    result = run(args, build_model=build_model, measure_model=measure_model,
                 train_epoch_fn=train_epoch, data_factory=data_factory)
    assert result["experiment"] == "cropr_qat" and result["use_cropr"]
    assert result["qat_scope"] == "all" and result["test_samples"] == 2
    assert result["token_pruning_ratio"] == 0.75


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("scope", ["all", "backbone"])
def test_cropr_qat_cuda(tmp_path, scope):
    args = settings(tmp_path, scope=scope)
    args.image_size, args.device, args.cropr_pruning_rate = 224, "cuda", 8
    model = build_model(args).cuda()
    images, labels = torch.randn(1, 3, 224, 224), torch.tensor([0])
    batches = DataLoader(TensorDataset(images, labels), batch_size=1)
    stats = train_epoch(model, batches, torch.optim.AdamW(model.parameters(), lr=args.lr), args, 0, 1)
    assert stats["train_aux_loss"] > 0
    metrics = measure_vit(model, 224, "cuda", 0, 1)
    assert metrics["token_pruning_ratio"] == 89 / 196
    assert metrics["latency_ms_mean"] > 0
