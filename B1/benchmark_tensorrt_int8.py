"""Export a B1 checkpoint as INT8 Q/DQ ONNX and time it against FP32 with TensorRT on the GPU.

The learned LSQ/LSQ+ grid is kept (see tensorrt_int8.py). The engines run on the GPU only; --export-only
writes the ONNX graphs on any machine. GPU compute time is measured with CUDA events at batch size 1.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from .config import CHECKPOINT_DEFAULTS, validate
from .metrics import measure_model
from .model import build_backbone, build_model, restore_model
from .modules import collapsed_quantizer_channels
from .tensorrt_engine import compare_logits, evaluate_fixed_batch, require_tensorrt, run_backend
from .tensorrt_int8 import (audit_quantized_products, convert_for_tensorrt, expected_quantized_products,
                            export_onnx)
from utils.classification import evaluate
from utils.data import loader, make_food101_data as make_data
from utils.reproducibility import configure_backend, seed_all

ENGINES = ("fp32_master", "b1_int8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="B1 best.pt or last.pt")
    parser.add_argument("--output", required=True, help="Directory for ONNX, engines, logs and reports")
    parser.add_argument("--batch-size", type=int, choices=[1], default=1, help="Timing batch is fixed to 1")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Accuracy evaluation batch size")
    parser.add_argument("--export-only", action="store_true", help="ONNX export only; no TensorRT required")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--data-root", help="Food-101 root; enables real-data accuracy evaluation")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup forwards on the single timing input")
    parser.add_argument("--iterations", type=int, default=100, help="Exact number of timed batch-1 forwards")
    options = parser.parse_args(argv)
    if options.eval_batch_size < 1 or options.warmup < 0 or options.iterations < 1:
        parser.error("eval-batch-size/iterations must be positive; warmup must be nonnegative")
    if options.workers < 0:
        parser.error("workers must be nonnegative")
    if options.export_only and options.data_root:
        parser.error("--data-root evaluates the engines; omit --export-only")
    if options.download and not options.data_root:
        parser.error("--download requires --data-root")
    return options


def evaluation_data(config, checkpoint, options):
    config = copy.copy(config)
    config.data_root, config.download = options.data_root, options.download
    config.batch_size, config.workers, config.device = options.eval_batch_size, options.workers, options.device
    _, val, test, manifest = make_data(config)
    if checkpoint.get("split_hash") != manifest["sha256"]:
        raise ValueError("Checkpoint data split does not match; use the original Food-101 data and split")
    dataset = test if options.eval_split == "test" else val
    return loader(dataset, config), manifest


def structure_metrics(model, config):
    """Architecture numbers per image; quantization does not change the arithmetic count."""
    config = copy.copy(config)
    config.device, config.latency_warmup, config.latency_repeats = "cpu", 0, 1
    measured = measure_model(model, config)
    fields = ("params", "inference_params", "gflops", "flops_convention", "token_pruning_ratio",
              "final_token_pruning_ratio", "pruning_ratio_definition", "tokens_per_block", "tokens_after_pruning")
    return {field: measured[field] for field in fields}


def write_metrics_csv(output, report):
    rows = []
    for name in ("source_qat", *ENGINES):
        if name in report:
            row = dict(backend=name, eval_split=report.get("eval_split"),
                       **report["model_structure"], **report[name])
            rows.append({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def load_models(options):
    """(config, checkpoint, trained fake-quant model, INT8 Q/DQ export model, FP32 master-weight model)."""
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    config = argparse.Namespace(**{**CHECKPOINT_DEFAULTS, **checkpoint["config"]})
    config.eval_checkpoint, config.resume = options.checkpoint, None
    validate(config)  # Requires a Linear QAT (mode linear_qat) B1 checkpoint.
    configure_backend(deterministic=True, allow_tf32=False)
    seed_all(config.seed)
    source = build_model(config)
    restore_model(source, checkpoint)  # Warns when the checkpoint has collapsed quantizer channels.
    source.eval()
    converted = convert_for_tensorrt(source)
    # The FP32 reference is the same checkpoint without quantization, not a separately trained model.
    fp32 = build_backbone(config)
    fp32.load_state_dict({key: checkpoint["model"][key] for key in fp32.state_dict()}, strict=True)
    return config, checkpoint, source, converted, fp32.eval()


def main(argv=None):
    options = parse_args(argv)
    trt = None if options.export_only else require_tensorrt(options.device)
    config, checkpoint, source, converted, fp32 = load_models(options)
    batches, manifest = evaluation_data(config, checkpoint, options) if options.data_root else (None, None)
    x = torch.randn(options.batch_size, 3, config.image_size, config.image_size)
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected, actual, references = source(x), converted(x), {"fp32_master": fp32(x)}
        references["b1_int8"] = actual
    report = dict(checkpoint=str(options.checkpoint), torch_version=str(torch.__version__),
                  batch_size=1, timing_batch_size=1,
                  eval_batch_size=options.eval_batch_size if batches is not None else None,
                  image_size=config.image_size,
                  weight_bits=config.weight_bits, activation_bits=config.activation_bits,
                  first_last_bits=config.first_last_bits,
                  cropr_scorer_bits=config.cropr_scorer_bits if config.use_cropr else None,
                  weight_quantizer=config.weight_quantizer, activation_quantizer=config.activation_quantizer,
                  weight_values="trained LSQ weight codes in an INT8 container (no packed 2/4-bit kernels)",
                  activations="trained activation grid unchanged: the offset is folded into the bias and "
                              "8-bit codes are shifted by 128 (zero point 0); no calibration or remapping",
                  fp32_ops="attention QK^T/AV, softmax, LayerNorm, GELU, residual adds, biases",
                  collapsed_quantizer_channels=collapsed_quantizer_channels(source),
                  synthetic_logit_max_abs_error=(expected - actual).abs().max().item(),
                  synthetic_top1_agreement=(expected.argmax(-1) == actual.argmax(-1)).float().mean().item(),
                  model_structure=structure_metrics(source, config),
                  structure_note="Source architecture, per image, not measured TensorRT kernel FLOPs; "
                                 "quantization does not reduce arithmetic count",
                  accuracy_evaluation="not_requested" if batches is None else "pending",
                  engine_execution="not_run", kernel_precision_verified=False,
                  note="Synthetic checks are not Food-101 accuracy. Q/DQ coverage is not proof of INT8 kernels: "
                       "inspect b1_int8.layers.json. GPU compute time excludes build, transfers and preprocessing.")
    for name, net in (("fp32_master", fp32), ("b1_int8", converted)):
        graph = export_onnx(net, x, output / f"{name}.onnx")
        if name == "b1_int8":
            products = audit_quantized_products(graph)
            quantized, expected_count = sum(row["quantized"] for row in products), expected_quantized_products(converted)
            if quantized != expected_count:
                raise RuntimeError(f"Expected {expected_count} Q/DQ products in the graph, found {quantized}: {products}")
            report.update(onnx_products=len(products), onnx_quantized_products=quantized)
    # Both engines use identical input bytes.
    x.numpy().tofile(output / "images.bin")
    report_path = output / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    if batches is not None:
        (output / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
        report.update(eval_split=options.eval_split, split_hash=manifest["sha256"], data_root=str(options.data_root),
                      accuracy_note="FP32 master shares QAT master weights; not a separately trained FP32 baseline")
        print("Evaluating the original fake-quant model on the same data", flush=True)
        source_metrics = evaluate(source.to(options.device), batches, options.device)
        report["source_qat"] = dict(top1_acc=source_metrics["top1_acc"], loss=source_metrics["loss"],
                                    evaluated_samples=source_metrics["samples"], eval_batch_size=options.eval_batch_size)
        source.cpu()
        with torch.cuda.device(options.device):
            torch.cuda.empty_cache()
        save()
    if not options.export_only:
        report.update(backend="tensorrt_python", backend_version=trt.__version__,
                      device=torch.cuda.get_device_name(options.device), cuda_version=torch.version.cuda,
                      tf32=False, amp=False, cuda_graph=False,
                      timing="CUDA events, synchronized each forward; excludes build and transfers",
                      warmup=options.warmup, iterations=options.iterations)
        for name in ENGINES:
            report["engine_execution"] = f"building_or_running_{name}"
            save()
            print(f"Building/benchmarking {name} with the TensorRT Python API", flush=True)
            try:
                net = fp32 if name == "fp32_master" else converted
                report[name] = run_backend(trt, output, name, net, x, references[name], options, batches,
                                           export_model=export_onnx, evaluate_batches=evaluate_fixed_batch,
                                           compare_outputs=compare_logits, allow_tf32=False,
                                           int8=name == "b1_int8")
                save()
            except Exception as error:
                report.update(engine_execution=f"failed_{name}", error=str(error))
                save()
                raise
        report["gpu_compute_speedup_vs_trt_fp32"] = (
            report["fp32_master"]["gpu_compute_ms_median"] / report["b1_int8"]["gpu_compute_ms_median"])
        report["engine_size_ratio_fp32_to_int8"] = (
            report["fp32_master"]["storage_engine_bytes"] / report["b1_int8"]["storage_engine_bytes"])
        if batches is not None:
            counts = [report[name]["evaluated_samples"] for name in ("source_qat", *ENGINES)]
            if any(count != len(batches.dataset) for count in counts):
                raise RuntimeError("Accuracy evaluation did not cover the complete selected split")
            report["int8_top1_drop_vs_source_qat_pp"] = report["source_qat"]["top1_acc"] - report["b1_int8"]["top1_acc"]
            report["int8_top1_drop_vs_trt_fp32_pp"] = report["fp32_master"]["top1_acc"] - report["b1_int8"]["top1_acc"]
            report["accuracy_evaluation"] = "completed"
        report["engine_execution"] = "completed; inspect b1_int8.layers.json to confirm INT8 kernels"
        save()
        write_metrics_csv(output, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
