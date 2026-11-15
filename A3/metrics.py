"""A3 paired inference measurement with CLS router and local QAT accounting."""
from utils.metrics import measure_vit as measure_base
from .pruning import AttentionRouter
from .quantization import storage_metrics


def measure_vit(model, image_size, device, warmup=10, repeats=100, *, sample=None):
    metrics = measure_base(model, image_size, device, warmup, repeats, sample=sample,
                           pruner_types=(AttentionRouter,))
    tokens = metrics["tokens_per_block"][0]
    patch_tokens = tokens - model.num_prefix_tokens
    minimum_tokens = min(metrics["tokens_after_pruning"], default=tokens)
    metrics.update(inference_params=metrics["params"],
                   storage_estimated_packed_bytes=metrics.pop("parameter_bytes"),
                   token_pruning_ratio=(tokens - minimum_tokens) / patch_tokens,
                   final_token_pruning_ratio=(tokens - metrics["tokens_per_block"][-1]) / patch_tokens,
                   pruning_ratio_definition="Maximum fraction of patch tokens removed before LLF; excludes CLS",
                   storage_estimate_note="Parameter payload and quantization metadata estimate; not an exported integer model",
                   integer_latency_ms=None)
    metrics.update(storage_metrics(model))
    metrics["router_params"] = sum(p.numel() for m in model.modules() if isinstance(m, AttentionRouter)
                                   for p in m.parameters())
    metrics["flops_convention"] = "2 FLOPs/MAC; inference Conv/Linear, ViT QK/AV; CLS score reduction excluded; excludes bias, norm, softmax, sorting, quantization"
    return metrics
