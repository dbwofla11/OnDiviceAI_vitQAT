"""Small real-GPU checks; skipped on machines without CUDA."""
import pytest
import torch

from tests.support import parser
from tests.support import build_model
from tests.support import measure_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("mode,weight_bits,activation_bits", [
    ("fp32", 2, 2), ("qat", 2, 2), ("qat", 4, 4), ("qat", 4, 8),
])
def test_cuda_training_step_and_metrics(mode, weight_bits, activation_bits):
    args = parser().parse_args([
        "--mode", mode, "--output", "runs/cuda-check", "--device", "cuda",
        "--weight-bits", str(weight_bits), "--activation-bits", str(activation_bits),
        "--latency-warmup", "0", "--latency-repeats", "1",
    ])
    model = build_model(args).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(1, 3, 224, 224, device="cuda")
    targets = torch.tensor([3], device="cuda")
    loss = torch.nn.functional.cross_entropy(model(images), targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metrics = measure_model(model, args)
    assert metrics["latency_ms_mean"] > 0
    assert metrics["latency_backend"] == ("fp32" if mode == "fp32" else "fake_quant_fp32")
    assert metrics["tokens_per_block"] == [197] * 12
