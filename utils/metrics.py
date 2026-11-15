"""ViT shape/FLOP counters and latency measurement; no model construction or precision policy."""
import math
import statistics
import time

import torch
from torch import nn



@torch.inference_mode()
def measure_vit(model, image_size, device, warmup=10, repeats=100, *, sample=None,
                pruner_types=(), scorer_flops=None):
    """Measure an explicit ViT contract (blocks, embed_dim and optional caller-supplied pruners)."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 are required")
    model.eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device) if sample is None else sample
    if x.shape != (1, 3, image_size, image_size) or x.device != torch.device(device):
        # cuda and cuda:0 refer to the same current device.
        expected_device = torch.empty(0, device=device).device
        if x.shape != (1, 3, image_size, image_size) or x.device != expected_device:
            raise ValueError("Latency sample must have shape [1,3,image_size,image_size] on the evaluation device")
    flops = 0
    tokens_per_block, tokens_after_pruning = [], []

    def count_attention(module, inputs):
        nonlocal flops
        n = inputs[0].shape[1]
        tokens_per_block.append(n)
        flops += 4 * n**2 * model.embed_dim

    def count_pruner(module, inputs, output):
        nonlocal flops
        # The caller supplies any extra operations performed by its pruners.
        if scorer_flops is not None:
            flops += scorer_flops(module, inputs)
        tokens_after_pruning.append(output[0].shape[1])

    def count(module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Linear):
            flops += 2 * output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            flops += 2 * output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    hooks += [block.register_forward_pre_hook(count_attention) for block in model.blocks]
    hooks += [m.register_forward_hook(count_pruner) for m in model.modules() if isinstance(m, pruner_types)]
    try:
        model(x)
    finally:
        for handle in hooks:
            handle.remove()
    params = sum(p.numel() for p in model.parameters())
    estimated_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(x)
    sync()
    timings = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        model(x)
        sync()
        timings.append((time.perf_counter() - start) * 1000)
    ordered = sorted(timings)
    return dict(
        params=params, gflops=flops / 1e9,
        flops_convention="2 FLOPs/MAC; inference Conv/Linear, ViT QK/AV and supplied scorer costs; excludes bias, norm, softmax, sorting, quantization",
        parameter_bytes=estimated_bytes,
        tokens_per_block=tokens_per_block, tokens_after_pruning=tokens_after_pruning,
        latency_ms_mean=statistics.mean(timings), latency_ms_median=statistics.median(timings),
        latency_ms_p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
        latency_batch_size=1, latency_warmup=warmup, latency_repeats=repeats,
        latency_scope="model forward only; input already on device; no data loading or transfer",
    )
