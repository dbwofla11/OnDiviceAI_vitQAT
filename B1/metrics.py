"""B1 measurement policy and experiment metadata."""
import math
import platform
import timm
import torch
import torchvision
from utils.results import save_results
from utils.metrics import measure_latency, measure_vit
from .cropr_adapter import Cropr, CrossAttention
from .integer import BACKEND, to_integer, unsupported_reason
from .modules import collapsed_quantizer_channels, quantized_modules, quantizers
from .quantizers import ActivationQuantizer


def quantization_summary(model):
    """Parameter counts and packed-storage estimate for Linear QAT, excluding quantizer parameters.

    utils.metrics.measure_vit counts LSQ step sizes as model parameters and reports raw FP32 bytes,
    so these fields replace the ones it reports.
    """
    quantizer_list = quantizers(model)
    quantizer_ids = {id(p) for q in quantizer_list for p in q.parameters()}
    auxiliary_ids = auxiliary_parameter_ids(model)
    layers = quantized_modules(model)
    packed_bits = {id(weight): quantizer.bits for _, weight, quantizer in layers}
    storage = inference_storage = 0
    for parameter in model.parameters():
        if id(parameter) in quantizer_ids:
            continue
        bits = packed_bits.get(id(parameter))
        size = math.ceil(parameter.numel() * bits / 8) if bits else parameter.numel() * parameter.element_size()
        storage += size
        inference_storage += 0 if id(parameter) in auxiliary_ids else size
    for quantizer in quantizer_list:
        if isinstance(quantizer, ActivationQuantizer):
            size = 8 if quantizer.asymmetric else 4  # FP32 scale (+ FP32 offset folded into bias)
        else:
            size = 4 * quantizer.channels  # FP32 scale per output channel, zero point fixed at 0
        storage += size
        inference_storage += size  # All quantizers are on the inference path.
    return dict(
        quantizer_params=sum(p.numel() for q in quantizer_list for p in q.parameters()),
        quantized_layers=[name for name, _, _ in layers],
        quantized_weight_params=sum(weight.numel() for _, weight, _ in layers),
        storage_estimated_packed_bytes=storage,
        inference_storage_estimated_packed_bytes=inference_storage,
        storage_estimate_note="Packed quantized Linear/Conv/scorer weights + FP32 for all other parameters "
                              "+ FP32 per-channel weight scales and per-tensor activation scale/offset; "
                              "storage_* includes training-only Cropr branches, inference_storage_* excludes "
                              "them; both exclude serialization overhead",
    )


def auxiliary_parameter_ids(model):
    """Training-only Cropr aggregator and auxiliary heads, which inference does not run."""
    ids = set()
    for module in model.modules():
        if isinstance(module, Cropr) and hasattr(module, "head"):
            ids.update(id(p) for p in module.head.parameters())
        if isinstance(module, CrossAttention):
            for name in ("v", "proj", "mlp_norm", "mlp"):
                branch = getattr(module, name, None)
                if branch is not None:
                    ids.update(id(p) for p in branch.parameters())
    return ids


def integer_latency(model, args):
    """Latency of the same quantization grids on integer (INT8) kernels; None fields carry the reason.

    The kernels are CPU only. Accuracy and the other fields still come from the fake-quant model.
    """
    fields = dict(integer_latency_ms=None, integer_latency_ms_mean=None, integer_latency_ms_p95=None,
                  integer_latency_backend=None, integer_latency_note=None)
    if not str(args.device).startswith("cpu"):
        fields["integer_latency_note"] = "PyTorch has no CUDA INT8 kernel; evaluate with --device cpu"
        return fields
    reason = unsupported_reason(model)
    if reason:
        fields["integer_latency_note"] = reason
        return fields
    integer_model = to_integer(model)
    result = measure_latency(integer_model, torch.zeros(1, 3, args.image_size, args.image_size), "cpu",
                             args.latency_warmup, args.latency_repeats)
    fields.update(
        integer_latency_ms=result["median"], integer_latency_ms_mean=result["mean"],
        integer_latency_ms_p95=result["p95"], integer_latency_backend=BACKEND,
        integer_latency_note="CPU INT8 kernels for quantized Linear/Conv/scorer; FP32 for attention matmuls, "
                             "softmax, norm, GELU and residual; weights of any bit width stored in INT8")
    return fields


def measure_model(model, args):
    metrics = measure_vit(model, args.image_size, args.device, args.latency_warmup, args.latency_repeats, pruner_types=(Cropr,),
                          scorer_flops=lambda stage, inputs: 2 * stage.num_queries * inputs[0].shape[1] * stage.embed_dim)
    tokens = metrics["tokens_per_block"][0]
    patch_tokens = tokens - model.num_prefix_tokens
    minimum_tokens = min(metrics["tokens_after_pruning"], default=tokens)
    summary = quantization_summary(model)
    auxiliary_ids = auxiliary_parameter_ids(model)
    # All quantizer parameters sit on the inference path (aux branches stay FP32).
    params = metrics["params"] - summary["quantizer_params"]
    auxiliary = sum(p.numel() for p in model.parameters() if id(p) in auxiliary_ids)
    metrics.pop("parameter_bytes")  # Raw FP32 bytes; replaced by the packed storage estimate.
    metrics.update(summary)
    metrics.update(params=params, inference_params=params - auxiliary,
                   token_pruning_ratio=(tokens - minimum_tokens) / patch_tokens,
                   final_token_pruning_ratio=(tokens - metrics["tokens_per_block"][-1]) / patch_tokens,
                   pruning_ratio_definition="Maximum fraction of patch tokens removed before LLF; excludes CLS",
                   latency_backend="fake_quant_fp32",
                   collapsed_quantizer_channels=collapsed_quantizer_channels(model))
    metrics.update(integer_latency(model, args))
    initialization = ("init_checkpoint" if getattr(args, "init_checkpoint", None)
                      else "timm_pretrained" if getattr(args, "pretrained", False) else "scratch")
    metrics.update(model=args.model, mode='linear_qat', use_cropr=args.use_cropr, cls_pruning=None,
                   cls_pruning_rate=0,
                   cropr_pruning_rate=args.cropr_pruning_rate if args.use_cropr else 0,
                   cropr_llf=args.cropr_llf if args.use_cropr else False,
                   weight_bits=args.weight_bits, activation_bits=args.activation_bits,
                   first_last_bits=args.first_last_bits,
                   cropr_scorer_bits=args.cropr_scorer_bits if args.use_cropr else None,
                   weight_quantizer=args.weight_quantizer, activation_quantizer=args.activation_quantizer,
                   always_fp32_ops="attention QK^T/AV matmuls, softmax, LayerNorm, GELU, residual, embeddings, "
                                   "Cropr aux branches; quantized_layers lists every quantized layer",
                   initialization=initialization,
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
