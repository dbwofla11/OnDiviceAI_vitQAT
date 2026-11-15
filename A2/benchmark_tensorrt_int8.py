"""Export and benchmark the QAT baseline checkpoint."""
import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from .config import CHECKPOINT_DEFAULTS, validate
from .model import restore_model, build_model, build_backbone
from .metrics import measure_model
from .tensorrt_int8 import audit_attention_qdq, convert_tensorrt_int8, export_onnx
from utils.tensorrt_benchmark import require_tensorrt, run_backend
from utils.data import make_food101_data as make_data, loader
from utils.classification import evaluate, evaluate_fixed_batch, compare_logits
from utils.reproducibility import seed_all, configure_backend


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
    config = copy.copy(config)
    config.device, config.latency_warmup, config.latency_repeats = "cpu", 0, 1
    measured = measure_model(model, config)
    fields = ("params", "inference_params", "gflops", "flops_convention", "token_pruning_ratio",
              "final_token_pruning_ratio", "pruning_ratio_definition", "tokens_per_block", "tokens_after_pruning")
    return {field: measured[field] for field in fields}


def write_metrics_csv(output, report):
    rows = []
    for name in ("source_qat", "fp32_master", "w2_int8"):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Directory for ONNX, engines, logs and reports")
    parser.add_argument("--batch-size", type=int, choices=[1], default=1, help="Timing batch is fixed to 1")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Accuracy evaluation batch size")
    parser.add_argument("--export-only", action="store_true", help="CPU export; no TensorRT required")
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
        parser.error("--data-root evaluates native engines; omit --export-only")
    if options.download and not options.data_root:
        parser.error("--download requires --data-root")
    trt = None if options.export_only else require_tensorrt(options.device)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    config = argparse.Namespace(**{**CHECKPOINT_DEFAULTS, **checkpoint["config"]})
    config.eval_checkpoint, config.resume = options.checkpoint, None
    validate(config)
    if config.mode != "qat" or config.use_cropr or config.cls_pruning or (config.weight_bits, config.activation_bits) != (2, 8):
        parser.error("A calibrated W2A8 QAT checkpoint is required")
    batches, manifest = evaluation_data(config, checkpoint, options) if options.data_root else (None, None)
    configure_backend(deterministic=True, allow_tf32=False)
    seed_all(config.seed)
    source = build_model(config)
    restore_model(source, checkpoint)
    source.eval()
    converted = convert_tensorrt_int8(source)
    fp32 = build_backbone(config)
    fp32.load_state_dict({key: checkpoint["model"][key] for key in fp32.state_dict()}, strict=True)
    fp32.eval()
    x = torch.randn(options.batch_size, 3, config.image_size, config.image_size)
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected, actual = source(x), converted(x)
        references = {"fp32_master": fp32(x), "w2_int8": actual}
    report = dict(checkpoint=str(options.checkpoint), torch_version=str(torch.__version__),
                  batch_size=1, timing_batch_size=1,
                  eval_batch_size=options.eval_batch_size if batches is not None else None,
                  image_size=config.image_size,
                  weight_values="original W2 grid exported with INT8 Q/DQ; kernel precision unverified",
                  activations="symmetric INT8 derived from trained affine range; no new calibration",
                  qat_scope=config.qat_scope,
                  synthetic_logit_max_abs_error=(expected - actual).abs().max().item(),
                  synthetic_top1_agreement=(expected.argmax(-1) == actual.argmax(-1)).float().mean().item(),
                  model_structure=structure_metrics(source, config),
                  structure_note="Source architecture, per image, not measured TensorRT kernel FLOPs; quantization does not reduce arithmetic count",
                  accuracy_evaluation="not_requested" if batches is None else "pending",
                  engine_execution="not_run", kernel_precision_verified=False,
                  note="Synthetic checks are not Food-101 accuracy. Q/DQ coverage is not proof of INT8 kernels.")
    for name, net in (("fp32_master", fp32), ("w2_int8", converted)):
        graph = export_onnx(net, x, output / f"{name}.onnx")
        if name == "w2_int8":
            rows = audit_attention_qdq(graph)
            expected_count = 2 * len(source.blocks)
            quantized = [row for row in rows if row["both_inputs_qdq"]]
            if len(quantized) != expected_count:
                raise RuntimeError(f"Expected {expected_count} Q/DQ attention BMMs, found {len(quantized)}: {rows}")
            report["attention_onnx"] = rows
    # Use identical actual input bytes for both native engines.
    x.numpy().tofile(output / "images.bin")
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if batches is not None:
        (output / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
        report.update(eval_split=options.eval_split, split_hash=manifest["sha256"],
                      data_root=str(options.data_root),
                      accuracy_note="FP32 master shares QAT master weights; not a separately trained FP32 baseline")
        print("Evaluating original affine QAT on the same data", flush=True)
        source_metrics = evaluate(source.to(options.device), batches, options.device)
        report["source_qat"] = dict(top1_acc=source_metrics["top1_acc"], loss=source_metrics["loss"],
                                    evaluated_samples=source_metrics["samples"], eval_batch_size=options.eval_batch_size)
        source.cpu()
        with torch.cuda.device(options.device):
            torch.cuda.empty_cache()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not options.export_only:
        report.update(backend="tensorrt_python", backend_version=trt.__version__,
                      device=torch.cuda.get_device_name(options.device), cuda_version=torch.version.cuda,
                      tf32=False, amp=False, cuda_graph=False,
                      timing="CUDA events, synchronized each forward; excludes build and transfers",
                      warmup=options.warmup, iterations=options.iterations)
        for name in ("fp32_master", "w2_int8"):
            report["engine_execution"] = f"building_or_running_{name}"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Building/benchmarking {name} with TensorRT Python API", flush=True)
            try:
                net = fp32 if name == "fp32_master" else converted
                report[name] = run_backend(trt, output, name, net, x, references[name], options, batches, export_model=export_onnx,
                                          evaluate_batches=evaluate_fixed_batch, compare_outputs=compare_logits, allow_tf32=False)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            except Exception as error:
                report["engine_execution"] = f"failed_{name}"
                report["error"] = str(error)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                raise
        report["gpu_compute_speedup_vs_trt_fp32"] = (
            report["fp32_master"]["gpu_compute_ms_median"] / report["w2_int8"]["gpu_compute_ms_median"])
        report["engine_size_ratio_fp32_to_int8"] = (
            report["fp32_master"]["storage_engine_bytes"] / report["w2_int8"]["storage_engine_bytes"])
        if batches is not None:
            counts = [report[name]["evaluated_samples"] for name in ("source_qat", "fp32_master", "w2_int8")]
            if any(count != len(batches.dataset) for count in counts):
                raise RuntimeError("Accuracy evaluation did not cover the complete selected split")
            report["int8_top1_drop_vs_source_qat_pp"] = report["source_qat"]["top1_acc"] - report["w2_int8"]["top1_acc"]
            report["int8_top1_drop_vs_trt_fp32_pp"] = report["fp32_master"]["top1_acc"] - report["w2_int8"]["top1_acc"]
            report["accuracy_evaluation"] = "completed"
        report["engine_execution"] = "completed; inspect layer reports to confirm INT8 attention kernels"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_metrics_csv(output, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
