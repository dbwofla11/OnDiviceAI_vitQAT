"""Compare dense A2 and A3 CLS pruning in TensorRT, using one W2A8 checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import torch

from A3.eval_pruning import inherited_args, sha256_file, validate_qat_state
from A3.metrics import measure_vit
from A3.pruning import attach_pruning
from A3.pruning.checkpoint import checkpoint_model, pruning_config
from scripts.benchmark_tensorrt_int8 import evaluation_data, require_tensorrt, run_backend
from vitqat.models import build_model
from vitqat.quantization import Quantizer
from vitqat.tensorrt_int8 import audit_attention_qdq, convert_tensorrt_int8, export_onnx
from vitqat.training import evaluate, seed_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New empty output directory")
    parser.add_argument("--data-root", help="Override checkpoint Food-101 root")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args(argv)
    options.latency_warmup, options.latency_repeats = options.warmup, options.iterations
    options.download = False
    output = Path(options.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose a new empty output directory")
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    config = inherited_args(checkpoint, options)
    if (config.weight_bits, config.activation_bits) != (2, 8):
        parser.error("TensorRT conversion currently requires an A2 W2A8 checkpoint")
    if options.threads:
        torch.set_num_threads(options.threads)
    trt = None if options.export_only else require_tensorrt(options.device)
    options.data_root = options.data_root or config.data_root
    batches, manifest = (None, None) if options.export_only else evaluation_data(config, checkpoint, options)
    seed_all(config.seed)
    source = build_model(config).cpu().eval()
    checkpoint_model(source, config).load_state_dict(checkpoint["model"], strict=True)
    source.eval()
    validate_qat_state(source)
    for module in source.modules():
        if isinstance(module, Quantizer):
            module.frozen = True
            module.fake_quant.disable_observer()
    source.requires_grad_(False)
    dense = convert_tensorrt_int8(source)
    source_pruned = attach_pruning(source, config=pruning_config(config)).eval()
    pruned = attach_pruning(dense, config=pruning_config(config)).eval()
    # Real evaluation sample for native timing; synthetic input for export smoke checks only.
    x = (torch.randn(1, 3, config.image_size, config.image_size) if batches is None
         else batches.dataset[0][0].unsqueeze(0))
    report = dict(checkpoint=str(options.checkpoint), checkpoint_sha256=sha256_file(options.checkpoint),
                  training_config=checkpoint["config"], options=vars(options),
                  torch_version=str(torch.__version__), engine_execution="not_run",
                  kernel_precision_verified=False, timing_batch_size=1,
                  timing_input="synthetic" if batches is None else f"{options.eval_split}[0]",
                  pruning=pruned.config.describe(len(dense.blocks), dense.patch_embed.num_patches),
                  note="Both variants use the same checkpoint weights; dense disables inference pruning, including for pruning-trained checkpoints. W2 values in INT8 Q/DQ, symmetric A8 remapping; no INT2 kernels. Inspect layer reports for actual INT8 kernels.",
                  results={})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if manifest is not None:
        report["split_hash"] = manifest["sha256"]
        (output / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
        report.update(backend_version=trt.__version__, device=torch.cuda.get_device_name(options.device),
                      cuda_version=torch.version.cuda, tf32=False, amp=False, cuda_graph=False)
    save()
    try:
        for name, original, converted in (("dense_int8", source, dense), ("a3_int8", source_pruned, pruned)):
            print(f"Exporting {name}", flush=True)
            measured = measure_vit(original, config.image_size, "cpu", warmup=0, repeats=1, sample=x)
            fields = ("params", "inference_params", "gflops", "flops_convention", "token_pruning_ratio",
                      "final_token_pruning_ratio", "tokens_per_block", "tokens_after_pruning")
            row = report["results"][name] = {key: measured[key] for key in fields}
            with torch.inference_mode():
                reference, original_logits = converted(x), original(x)
            row["conversion_sample_max_abs_error"] = (reference - original_logits).abs().max().item()
            graph = export_onnx(converted, x, output / f"{name}.onnx")
            attention = audit_attention_qdq(graph)
            if len(attention) != 2 * len(dense.blocks) or not all(item["both_inputs_qdq"] for item in attention):
                raise RuntimeError(f"Incomplete attention Q/DQ coverage: {attention}")
            row["attention_onnx"] = attention
            save()
            if trt is not None:
                report["engine_execution"] = f"running_{name}"
                save()
                original.to(options.device)
                metrics = evaluate(original, batches, options.device)
                if metrics["samples"] != len(batches.dataset):
                    raise RuntimeError("Incomplete source QAT accuracy evaluation")
                row["source_qat_top1_acc"] = metrics["top1_acc"]
                row["source_qat_loss"] = metrics["loss"]
                row["source_qat_samples"] = metrics["samples"]
                original.cpu()
                with torch.cuda.device(options.device):
                    torch.cuda.empty_cache()
                row.update(run_backend(trt, output, name, converted, x, reference, options, batches))
                if row["evaluated_samples"] != len(batches.dataset):
                    raise RuntimeError("Incomplete accuracy evaluation")
                row["conversion_top1_drop_pp"] = row["source_qat_top1_acc"] - row["top1_acc"]
                save()
        if trt is not None:
            d, p = report["results"]["dense_int8"], report["results"]["a3_int8"]
            report["comparison"] = dict(top1_drop_pp=d["top1_acc"] - p["top1_acc"],
                latency_speedup=d["gpu_compute_ms_median"] / p["gpu_compute_ms_median"])
            report["engine_execution"] = "completed; kernel precision requires layer inspection"
        rows = [dict(condition=name, **{k: json.dumps(v) if isinstance(v, list) else v
                                       for k, v in row.items() if k != "attention_onnx"})
                for name, row in report["results"].items()]
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    except Exception as error:
        report.update(engine_execution="failed", error=str(error))
        raise
    finally:
        save()
    print(json.dumps(report.get("comparison", {"engine_execution": report["engine_execution"]}), indent=2))
    return report


if __name__ == "__main__":
    main()
