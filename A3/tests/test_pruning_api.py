import copy

import pytest
import torch

from A3.pruning import attach_pruning, PrunedOutput
from A3.tests.test_inference_pruning import settings, small_fixture_model, fixture_queries


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_inference_api_dense_train_pruned_eval(tmp_path):
    backbone = small_fixture_model(settings(tmp_path, "fp32"))
    keys = set(backbone.state_dict())
    params = {k: v.clone() for k, v in backbone.named_parameters()}
    wrapped = attach_pruning(backbone)
    assert wrapped.training and backbone.training
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(wrapped(x), backbone(x))
    assert set(backbone.state_dict()) == keys
    assert all(torch.equal(p, params[k]) for k, p in backbone.named_parameters())
    assert all(not p.requires_grad for p in wrapped.routers.parameters())
    wrapped.eval()
    wrapped.capture_trace = True
    assert wrapped(x).shape == (2, 101)
    assert wrapped.last_trace["tokens_per_block"][-2:] == [36, 197]
    wrapped.enabled = False
    assert torch.equal(wrapped(x), backbone(x))
    with pytest.raises(ValueError, match="already attached"):
        attach_pruning(wrapped)


@pytest.mark.parametrize("mode", ["fp32", "qat"])
def test_train_only_api_pruned_train_dense_eval(tmp_path, mode):
    backbone = small_fixture_model(settings(tmp_path, mode))
    model = attach_pruning(backbone, apply_during="train")
    model.capture_trace = True
    images = torch.randn(2, 3, 224, 224)
    model.train()
    logits = model(images)
    assert model.last_trace["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert backbone.patch_embed.proj.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(images), backbone(images))
    assert model.last_trace is None
    with pytest.raises(ValueError, match="does not use queries"):
        attach_pruning(small_fixture_model(settings(tmp_path, mode)), queries=fixture_queries(),
                       apply_during="train")


def test_train_api_gradients_export_reload_and_loss(tmp_path):
    backbone = small_fixture_model(settings(tmp_path, "fp32"))
    wrapped = attach_pruning(backbone, apply_during="cropr_train_and_inference")
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([1, 2])
    out = wrapped(x)
    assert isinstance(out, PrunedOutput) and len(out.auxiliary_logits) == 10
    criterion = torch.nn.CrossEntropyLoss()
    criterion(out.logits, y).backward(retain_graph=True)
    assert wrapped.cropr[0].cross_attn.queries.grad is None
    optimizer.zero_grad(set_to_none=True)
    out.auxiliary_loss(y, criterion).backward()
    assert backbone.patch_embed.proj.weight.grad is None  # stop-gradient
    assert wrapped.cropr[0].cross_attn.queries.grad.abs().sum() > 0
    optimizer.step()
    state = copy.deepcopy(wrapped.state_dict())
    wrapped.eval()
    expected = wrapped(x)
    assert len(wrapped.export_queries()) == 10
    restored = attach_pruning(small_fixture_model(settings(tmp_path, "fp32")),
                             apply_during="cropr_train_and_inference")
    restored.load_state_dict(state, strict=True)
    assert torch.equal(restored.eval()(x), expected)


def test_errors_and_soft_target_loss(tmp_path):
    backbone = small_fixture_model(settings(tmp_path, "fp32"))
    with pytest.raises(ValueError, match="does not use queries"):
        attach_pruning(backbone, queries=fixture_queries())
    with pytest.raises(ValueError, match="apply_during"):
        attach_pruning(backbone, apply_during="bad")
    from timm.loss import SoftTargetCrossEntropy
    model = attach_pruning(backbone, apply_during="cropr_train_and_inference")
    target = torch.nn.functional.one_hot(torch.tensor([1, 2]), 101).float()
    out = model(torch.randn(2, 3, 224, 224))
    loss = SoftTargetCrossEntropy()(out.logits, target) + out.auxiliary_loss(target, SoftTargetCrossEntropy())
    loss.backward()
    assert torch.isfinite(loss)


def test_qat_train_api_optimizer_and_eval_state(tmp_path):
    from A3.quantization import Quantizer
    from A3.eval_pruning import state_digest
    backbone = small_fixture_model(settings(tmp_path, "qat"))
    model = attach_pruning(backbone, apply_during="cropr_train_and_inference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    query = model.cropr[0].cross_attn.queries
    before = query.detach().clone()
    out = model(torch.randn(2, 3, 224, 224))
    y = torch.tensor([1, 2])
    ce = torch.nn.CrossEntropyLoss()
    (ce(out.logits, y) + out.auxiliary_loss(y, ce)).backward()
    assert backbone.patch_embed.proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(query, before)
    model.eval()
    for module in backbone.modules():
        if isinstance(module, Quantizer):
            module.fake_quant.disable_observer()
    original = state_digest(model)
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 101) and torch.isfinite(logits).all()
    assert state_digest(model) == original


@pytest.mark.parametrize("mode", ["fp32", "qat"])
def test_cls_attention_training_and_inference(tmp_path, mode):
    backbone = small_fixture_model(settings(tmp_path, mode))
    model = attach_pruning(backbone, apply_during="train_and_inference")
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in backbone.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = backbone.patch_embed.proj.weight.detach().clone()
    images = torch.randn(2, 3, 224, 224)
    target = torch.tensor([1, 2])
    model.capture_trace = True
    attn = backbone.blocks[0].attn
    source = getattr(attn, "prob_quant", attn.attn_drop)
    attention = []
    hook = source.register_forward_hook(lambda module, inputs, output: attention.append(output.detach().clone()))
    try:
        model.train()
        logits = model(images)
    finally:
        hook.remove()
    assert isinstance(logits, torch.Tensor) and logits.shape == (2, 101)
    assert model.last_trace["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    scores = attention[0][:, :, 0, :].mean(dim=1)
    scores[:, 0] = torch.inf
    assert torch.equal(model.last_trace["removed_ids"][0], scores.argsort(dim=1, descending=True)[:, -17:])
    torch.nn.functional.cross_entropy(logits, target).backward()
    assert backbone.patch_embed.proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(backbone.patch_embed.proj.weight, before)
    model.eval()
    with torch.no_grad():
        eval_logits = model(images)
        assert eval_logits.shape == (2, 101)
        inference_model = attach_pruning(backbone).eval()
        assert torch.equal(eval_logits, inference_model(images))
    restored = attach_pruning(small_fixture_model(settings(tmp_path, mode)),
                              apply_during="train_and_inference")
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(eval_logits, restored(images))
    assert model.last_trace["tokens_per_block"][-2:] == [36, 197]
    model.enabled = False
    with torch.no_grad():
        assert torch.equal(model(images), backbone(images))
    with pytest.raises(ValueError, match="does not use queries"):
        attach_pruning(small_fixture_model(settings(tmp_path, mode)), queries=fixture_queries(),
                       apply_during="train_and_inference")
