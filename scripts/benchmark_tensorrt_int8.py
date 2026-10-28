"""Export W2-valued INT8 ONNX and benchmark with the TensorRT Python API."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import statistics

import torch

from vitqat.cli import parser as training_parser
from vitqat.models import build_model
from vitqat.data import loader, make_data
from vitqat.metrics import measure_vit
from vitqat.tensorrt_int8 import audit_attention_qdq, convert_tensorrt_int8, export_onnx
from vitqat.training import evaluate, seed_all


def summarize_results(times, logits, reference):
    times = sorted(times)
    if not times or not all(math.isfinite(value) and value > 0 for value in times):
        raise ValueError("Missing or invalid native GPU timings")
    if logits.shape != reference.shape or not torch.isfinite(logits).all():
        raise ValueError("Invalid native logits shape or nonfinite values")
    return dict(gpu_compute_ms_mean=statistics.mean(times), gpu_compute_ms_median=statistics.median(times),
                gpu_compute_ms_p95=times[math.ceil(len(times) * 0.95) - 1], samples=len(times),
                synthetic_native_vs_reference_max_abs_error=(logits - reference).abs().max().item(),
                synthetic_native_vs_reference_top1_agreement=(logits.argmax(-1) == reference.argmax(-1)).float().mean().item())


@torch.inference_mode()
def evaluate_fixed_batch(predict, batches, batch_size):
    """Evaluate a fixed-shape engine without dropping or counting padded samples."""
    loss, correct, samples = 0.0, 0, 0
    for images, target in batches:
        count = images.shape[0]
        if not 0 < count <= batch_size:
            raise ValueError("Evaluation batch must fit the fixed engine batch size")
        if count < batch_size:
            images = torch.cat([images, images.new_zeros((batch_size - count, *images.shape[1:]))])
        output = predict(images)
        if output.ndim != 2 or output.shape[0] != batch_size or not torch.isfinite(output).all():
            raise ValueError("Invalid classification logits from engine")
        output = output[:count]
        loss += torch.nn.functional.cross_entropy(output, target, reduction="sum").item()
        correct += (output.argmax(-1) == target).sum().item()
        samples += count
    if not samples:
        raise ValueError("Empty evaluation dataset")
    return dict(top1_acc=100.0 * correct / samples, loss=loss / samples, evaluated_samples=samples)


def evaluation_data(config, checkpoint, options):
    config = copy.copy(config)
    config.data_root, config.download = options.data_root, options.download
    config.batch_size, config.workers, config.device = options.eval_batch_size, options.workers, options.device
    _, val, test, manifest = make_data(config)
    if checkpoint.get("split_hash") != manifest["sha256"]:
        raise ValueError("Checkpoint data split does not match; use the original Food-101 data and split")
    dataset = test if options.eval_split == "test" else val
    return loader(dataset, config), manifest


def structure_metrics(model, image_size):
    measured = measure_vit(model, image_size, "cpu", warmup=0, repeats=1)
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


def require_tensorrt(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires CUDA-enabled PyTorch; use --export-only for CPU export")
    try:
        import tensorrt as trt
    except (ImportError, OSError) as error:
        raise RuntimeError("Install TensorRT with: python -m pip install -r requirements-tensorrt.txt") from error
    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This runner requires TensorRT 10.x; install requirements-tensorrt.txt")
    return trt


def time_forwards(launch, stream, warmup, iterations):
    launch()  # First execution and pending input transfer are not measured.
    stream.synchronize()
    for _ in range(warmup):
        launch()
        stream.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iterations):
        start.record(stream)
        launch()
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


@torch.inference_mode()
def benchmark_engine(trt, output, name, x, reference, options, batches=None, *, measure=True):
    """Build once, bind PyTorch CUDA buffers, then time serialized forwards."""
    caller_stream = torch.cuda.current_stream(options.device)
    # Copies, TensorRT enqueue and CUDA events must use the SAME non-default stream.
    with torch.cuda.device(options.device), torch.cuda.stream(torch.cuda.Stream(device=options.device)):
        stream = torch.cuda.current_stream()
        stream.wait_stream(caller_stream)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)  # TensorRT 10 always uses explicit batch.
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(output / f"{name}.onnx")):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"{name}: TensorRT ONNX parsing failed:\n{errors}")
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"{name}: TensorRT engine build failed; see TensorRT messages above")
        (output / f"{name}.engine").write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(f"{name}: TensorRT engine deserialization failed")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"{name}: cannot create execution context")
        buffers = {"images": x.to(options.device).contiguous(),
                   "logits": torch.empty(reference.shape, dtype=torch.float32, device=options.device)}
        if engine.num_io_tensors != 2:
            raise RuntimeError(f"{name}: expected images/logits I/O only")
        for key, tensor in buffers.items():
            expected_mode = trt.TensorIOMode.INPUT if key == "images" else trt.TensorIOMode.OUTPUT
            if (engine.get_tensor_dtype(key) != trt.float32
                    or engine.get_tensor_mode(key) != expected_mode
                    or engine.get_tensor_location(key) != trt.TensorLocation.DEVICE
                    or engine.get_tensor_format(key) != trt.TensorFormat.LINEAR
                    or tuple(engine.get_tensor_shape(key)) != tuple(tensor.shape)):
                raise RuntimeError(f"{name}: unexpected fixed FP32 I/O contract for {key}")
            if not context.set_tensor_address(key, tensor.data_ptr()):
                raise RuntimeError(f"{name}: could not bind {key}")
        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        (output / f"{name}.layers.json").write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8")
        def launch():
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"{name}: TensorRT execution failed")

        result = dict(cuda_stream="dedicated_non_default",
                      storage_engine_bytes=(output / f"{name}.engine").stat().st_size,
                      storage_onnx_bytes=(output / f"{name}.onnx").stat().st_size)
        if measure:
            # One fixed input batch, not the evaluation dataset. No duration floor.
            times = time_forwards(launch, stream, options.warmup, options.iterations)
            logits = buffers["logits"].cpu()
            (output / f"{name}.times.json").write_text(
                json.dumps([{"computeMs": value} for value in times]), encoding="utf-8")
            (output / f"{name}.outputs.json").write_text(json.dumps([
                {"name": "logits", "shape": list(logits.shape), "values": logits.flatten().tolist()}]), encoding="utf-8")
            result.update(summarize_results(times, logits, reference))
            result["compute_only_images_per_second"] = 1000 * x.shape[0] / result["gpu_compute_ms_mean"]
        if batches is not None:
            print(f"Evaluating {name} on real data (outside timing)", flush=True)

            def predict(images):
                buffers["images"].copy_(images)
                launch()
                return buffers["logits"].cpu()

            result.update(evaluate_fixed_batch(predict, batches, x.shape[0]))
        return result


def run_backend(trt, output, name, model, x, reference, options, batches):
    """Time batch 1 only; build an untimed accuracy engine when batch sizes differ."""
    shared = batches is not None and options.eval_batch_size == x.shape[0]
    result = benchmark_engine(trt, output, name, x, reference, options, batches if shared else None)
    result.update(timing_batch_size=x.shape[0], timing_engine_name=name)
    if batches is None:
        return result
    eval_name = name
    accuracy = result
    if not shared:
        eval_name = f"{name}_eval_b{options.eval_batch_size}"
        eval_x = x.expand(options.eval_batch_size, *x.shape[1:]).contiguous()
        # Untimed execution only needs the output shape, not synthetic reference values.
        eval_reference = reference.expand(options.eval_batch_size, *reference.shape[1:])
        print(f"Exporting/building accuracy engine {eval_name} (no timing pass)", flush=True)
        export_onnx(model, eval_x, output / f"{eval_name}.onnx")
        accuracy = benchmark_engine(trt, output, eval_name, eval_x, eval_reference, options, batches, measure=False)
        for key in ("top1_acc", "loss", "evaluated_samples"):
            result[key] = accuracy[key]
    result.update(eval_batch_size=options.eval_batch_size, eval_engine_name=eval_name,
                  eval_storage_engine_bytes=accuracy["storage_engine_bytes"],
                  eval_storage_onnx_bytes=accuracy["storage_onnx_bytes"])
    return result


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
    config = training_parser().parse_args(["--mode", "qat", "--output", "unused"])
    vars(config).update(checkpoint["config"])
    if config.mode != "qat" or (config.weight_bits, config.activation_bits) != (2, 8):
        parser.error("A calibrated W2A8 QAT checkpoint is required")
    batches, manifest = evaluation_data(config, checkpoint, options) if options.data_root else (None, None)
    seed_all(config.seed)
    source = build_model(config)
    source.load_state_dict(checkpoint["model"], strict=True)
    source.eval()
    converted = convert_tensorrt_int8(source)
    fp_config = copy.copy(config)
    fp_config.mode = "fp32"
    fp32 = build_model(fp_config)
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
                  model_structure=structure_metrics(source, config.image_size),
                  structure_note="Source architecture, per image, not measured TensorRT kernel FLOPs; quantization does not reduce arithmetic count",
                  accuracy_evaluation="not_requested" if batches is None else "pending",
                  engine_execution="not_run", kernel_precision_verified=False,
                  note="Synthetic checks are not Food-101 accuracy. Q/DQ coverage is not proof of INT8 kernels.")
    for name, net in (("fp32_master", fp32), ("w2_int8", converted)):
        graph = export_onnx(net, x, output / f"{name}.onnx")
        if name == "w2_int8":
            rows = audit_attention_qdq(graph)
            expected_count = 2 * len(source.blocks)
            if config.use_cropr and config.qat_scope == "all":
                expected_count += len(source.cropr)
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
                report[name] = run_backend(trt, output, name, net, x, references[name], options, batches)
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
