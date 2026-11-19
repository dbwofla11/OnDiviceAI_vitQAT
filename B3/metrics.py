"""B3 measurement policy and experiment metadata."""
import platform
import timm
import torch
import torchvision
from utils.results import save_results
from utils.metrics import measure_vit
from .quantization import storage_metrics
from .cropr_adapter import Cropr, CrossAttention


def measure_model(model, args):
    metrics = measure_vit(model, args.image_size, args.device, args.latency_warmup, args.latency_repeats, pruner_types=(Cropr,),
                          scorer_flops=lambda stage, inputs: 2 * stage.num_queries * inputs[0].shape[1] * stage.embed_dim)
    tokens = metrics["tokens_per_block"][0]
    patch_tokens = tokens - model.num_prefix_tokens
    minimum_tokens = min(metrics["tokens_after_pruning"], default=tokens)
    metrics.update(inference_params=metrics["params"],
                   token_pruning_ratio=(tokens - minimum_tokens) / patch_tokens,
                   final_token_pruning_ratio=(tokens - metrics["tokens_per_block"][-1]) / patch_tokens,
                   pruning_ratio_definition="Maximum fraction of patch tokens removed before LLF; excludes CLS",
                   storage_estimate_note="Estimated packed parameter payload including FP32 auxiliary weights and quantization metadata",
                   integer_latency_ms=None)
    metrics.update(storage_metrics(model))
    auxiliary = 0
    for module in model.modules():
        if isinstance(module, Cropr) and hasattr(module, "head"):
            auxiliary += sum(p.numel() for p in module.head.parameters())
        if isinstance(module, CrossAttention):
            for name in ("v", "proj", "mlp_norm", "mlp"):
                branch = getattr(module, name, None)
                if branch is not None:
                    auxiliary += sum(p.numel() for p in branch.parameters())
    metrics["inference_params"] -= auxiliary
    metrics.update(model=args.model, mode=args.mode, use_cropr=True, cls_pruning=None,
                   cls_pruning_rate=0,
                   cropr_pruning_rate=args.cropr_pruning_rate, cropr_llf=args.cropr_llf,
                   ptq_scope=args.ptq_scope if args.mode == "ptq" else None,
                   weight_bits=args.weight_bits if args.mode == "ptq" else 32,
                   activation_bits=args.activation_bits if args.mode == "ptq" else 32,
                   pretrained=getattr(args, "pretrained", False), amp=False, tf32=False)
    return metrics


def report_results(model, args, test, checkpoint, selected, manifest, output, metrics):
    """Own this experiment's report fields and PyTorch artifact format."""
    metrics = dict(metrics)
    best_score = checkpoint.get("best_score", checkpoint.get("best_acc"))
    if args.mode == "ptq":
        metrics.update(source_fp32_checkpoint=checkpoint["source_checkpoint"],
                       source_fp32_best_val_top1_acc=best_score,
                       calibration_samples=checkpoint["calibration"]["samples"])
    else:
        metrics["best_val_top1_acc"] = best_score
    torch.save(model.state_dict(), output / "model_state.pt")
    metrics.update(experiment=args.experiment,
                   top1_acc=test["top1_acc"], test_loss=test["loss"], test_samples=test["samples"],
                   selected_epoch=checkpoint["epoch"],
                   checkpoint=str(selected), split_hash=manifest["sha256"],
                   storage_model_state_bytes=(output / "model_state.pt").stat().st_size,
                   seed=args.seed, device=args.device,
                   device_name=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else platform.processor(),
                   torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                   timm_version=timm.__version__, python_version=platform.python_version(),
                   cpu_threads=torch.get_num_threads(), image_size=args.image_size)
    return save_results(metrics, output)
