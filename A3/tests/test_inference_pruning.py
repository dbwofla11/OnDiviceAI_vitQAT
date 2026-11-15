import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from tests.support import parser
from B2.cropr_adapter import adapt_model, CrossAttention
from A3.inference_pruning import InferencePrunedViT, PruningConfig, AttentionRouter
from A3.metrics import measure_vit
from A3.model import build_model
from A3.quantization import prepare_qat


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def settings(tmp_path, mode="qat"):
    return parser().parse_args([
        "--mode", mode, "--weight-bits", "4", "--activation-bits", "8",
        "--image-size", "224", "--workers", "0", "--device", "cpu",
        "--drop-path", "0", "--batch-size", "64", "--output", str(tmp_path / "training"),
    ])


def small_fixture_model(args):
    """Real 12-block adapter/QAT contract, narrow width for fast CPU tests."""
    base = VisionTransformer(img_size=args.image_size, patch_size=16, num_classes=101,
                             embed_dim=12, depth=12, num_heads=3, drop_path_rate=args.drop_path)
    model = adapt_model(base, use_cropr=args.use_cropr, pruning_rate=args.cropr_pruning_rate,
                        llf=args.cropr_llf, image_size=args.image_size, drop_path=args.drop_path)
    if args.mode == "qat":
        prepare_qat(model, args.weight_bits, args.activation_bits, args.qat_scope)
    return model


def fixture_queries(dim=12):
    # Synthetic test tensors only; never exported as a trained research router.
    return {str(i): torch.randn(1, 1, dim) for i in range(1, 11)}


def test_cls_attention_topk_per_image():
    x = torch.arange(2 * 4 * 2).reshape(2, 4, 2).float()
    attention = torch.zeros(2, 2, 4, 4)
    attention[0, :, 0, :] = torch.tensor([[0., 0.8, 0.1, 0.1], [0., 0.4, 0.3, 0.3]])
    attention[1, :, 0, :] = torch.tensor([[0., 0.1, 0.8, 0.1], [0., 0.2, 0.6, 0.2]])
    keep, dropped = AttentionRouter(2)(x, attention)
    assert torch.equal(keep[:, 0], x[:, 0])
    assert torch.equal(keep[0, 1], x[0, 1])
    assert torch.equal(keep[1, 1], x[1, 2])
    assert dropped.shape == (2, 2, 2)


@pytest.mark.parametrize("mode", ["fp32", "qat"])
def test_original_forward_equivalence_llf_and_metrics(tmp_path, mode):
    args = settings(tmp_path, mode)
    dense = small_fixture_model(args)
    images = torch.randn(2, 3, 224, 224)
    if mode == "qat":
        dense(images)  # Initialize synthetic fixture observers, not real task training.
    dense.eval()
    rng = torch.get_rng_state().clone()
    wrapper = InferencePrunedViT(dense)
    assert torch.equal(rng, torch.get_rng_state())
    wrapper.enabled = False
    assert torch.equal(wrapper(images), dense(images))
    wrapper.enabled = True
    wrapper.capture_trace = True
    captured = []
    attn = dense.blocks[0].attn
    source = getattr(attn, "prob_quant", attn.attn_drop)
    handle = source.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    try:
        assert wrapper(images).shape == (2, 101)
    finally:
        handle.remove()
    expected = captured[0][:, :, 0, :].mean(dim=1).clone()
    expected[:, 0] = torch.inf
    expected_removed = expected.argsort(dim=1, descending=True)[:, -17:]
    trace = wrapper.last_trace
    assert torch.equal(trace["removed_ids"][0], expected_removed)
    assert trace["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    assert torch.equal(trace["restored_ids"].sort(dim=1).values, torch.arange(197).expand(2, -1))
    assert trace["restored_ids"][:, 0].eq(0).all()
    assert sum(ids.shape[1] for ids in trace["removed_ids"]) == 161
    assert torch.equal(wrapper(images), wrapper(images))
    wrapper.capture_trace = False
    wrapper(images)
    assert wrapper.last_trace is None
    metrics = measure_vit(wrapper, 224, "cpu", 0, 1, sample=images[:1])
    dense_metrics = measure_vit(dense, 224, "cpu", 0, 1)
    assert metrics["token_pruning_ratio"] == 161 / 196
    assert metrics["final_token_pruning_ratio"] == 0
    assert metrics["router_params"] == 0
    assert metrics["params"] == dense_metrics["params"]
    assert metrics["inference_params"] == metrics["params"]
    ns = [197,180,164,148,132,116,100,84,68,52,36,197]
    d = 12
    expected_flops = 2 * 196 * d * (3 * 16 * 16) + 2 * d * 101
    expected_flops += sum(24 * n * d * d + 4 * n * n * d for n in ns)
    assert metrics["gflops"] == expected_flops / 1e9
    assert metrics["gflops"] < dense_metrics["gflops"]
    assert all(not p.requires_grad for p in wrapper.routers.parameters())
    with pytest.raises(RuntimeError, match="cannot train"):
        wrapper.train()(images)


def test_validation_no_random_or_invalid_schedule(tmp_path):
    model = small_fixture_model(settings(tmp_path, "fp32"))
    with pytest.raises(ValueError, match="retain"):
        PruningConfig().schedule(12, 16)
    with pytest.raises(ValueError, match="LLF"):
        PruningConfig(llf=False).schedule(12, 196)


def test_documented_small_w4a8_forward(tmp_path):
    args = settings(tmp_path)
    args.model = "vit_small_patch16_224"
    model = build_model(args)
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(image)  # Synthetic observer initialization; not task training.
        model.eval()
        pruned = InferencePrunedViT(model)
        pruned.capture_trace = True
        output = pruned(image)
    assert output.shape == (1, 101) and torch.isfinite(output).all()
    assert pruned.last_trace["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    assert sum(p.numel() for p in pruned.routers.parameters()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_pruned_qat_cuda(tmp_path):
    model = small_fixture_model(settings(tmp_path)).cuda()
    image = torch.randn(1, 3, 224, 224, device="cuda")
    model(image)
    model.eval()
    wrapper = InferencePrunedViT(model).cuda()
    measured = measure_vit(wrapper, 224, "cuda", 0, 1, sample=image)
    assert measured["token_pruning_ratio"] == 161 / 196
    assert measured["latency_ms_median"] > 0
