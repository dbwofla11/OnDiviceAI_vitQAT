"""A1 measurement policy and experiment metadata."""
import platform
import timm
import torch
import torchvision
from utils.results import save_results
from utils.metrics import measure_vit


def measure_model(model, args):
    metrics = measure_vit(model, args.image_size, args.device, args.latency_warmup, args.latency_repeats)
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
    metrics.update(quantized_weight_params=0, latency_backend="fp32")
    metrics.update(model=args.model, mode='fp32', use_cropr=False, cls_pruning=None,
                   cls_pruning_rate=0,
                   cropr_pruning_rate=0, cropr_llf=False,
                   qat_scope=args.qat_scope, weight_bits=32,
                   activation_bits=32,
                   pretrained=getattr(args, "pretrained", False), amp=False, tf32=False)
    return metrics


def report_results(model, args, test, checkpoint, selected, manifest, output, metrics):
    """Own this experiment's report fields and PyTorch artifact format."""
    metrics = dict(metrics)
    torch.save(model.state_dict(), output / "model_state.pt")
    metrics.update(experiment=args.experiment,
                   top1_acc=test["top1_acc"], test_loss=test["loss"], test_samples=test["samples"],
                   selected_epoch=checkpoint["epoch"], best_val_top1_acc=checkpoint.get("best_score", checkpoint.get("best_acc")),
                   checkpoint=str(selected), split_hash=manifest["sha256"],
                   storage_model_state_bytes=(output / "model_state.pt").stat().st_size,
                   seed=args.seed, device=args.device,
                   device_name=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else platform.processor(),
                   torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                   timm_version=timm.__version__, python_version=platform.python_version(),
                   cpu_threads=torch.get_num_threads(), image_size=args.image_size,
                   )
    return save_results(metrics, output)
