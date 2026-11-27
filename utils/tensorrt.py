"""TensorRT 10 engines for fixed FP32 images/logits I/O: build, time, evaluate; shared by every experiment.

deployment_latency() is what each experiment's evaluation calls: it exports the evaluated model (an INT8
Q/DQ copy from utils.int8 for QAT experiments, the model itself for FP32), builds a TensorRT engine and
times batch-1 forwards on the GPU, so latency does not include fake-quantization overhead.
"""
import copy
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from .int8 import export_onnx



def summarize_results(times, logits, reference):
    times = sorted(times)
    if not times or not all(math.isfinite(value) and value > 0 for value in times):
        raise ValueError("Missing or invalid native GPU timings")
    if logits.shape != reference.shape or not torch.isfinite(logits).all():
        raise ValueError("Invalid native logits shape or nonfinite values")
    return dict(gpu_compute_ms_mean=statistics.mean(times), gpu_compute_ms_median=statistics.median(times),
                gpu_compute_ms_p95=times[math.ceil(len(times) * 0.95) - 1], samples=len(times),
                synthetic_native_vs_reference_max_abs_error=(logits - reference).abs().max().item())


def require_tensorrt(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires CUDA-enabled PyTorch; use --export-only for CPU export")
    try:
        import tensorrt as trt
    except (ImportError, OSError) as error:
        raise RuntimeError("Install TensorRT with: python -m pip install -r requirements.txt") from error
    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This runner requires TensorRT 10.x; install requirements.txt")
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
def benchmark_engine(trt, output, name, x, reference, options, batches=None, *, evaluate_batches, compare_outputs,
                     allow_tf32, measure=True, int8=False):
    """Build once, bind PyTorch CUDA buffers, then time serialized forwards.

    int8 enables the INT8 builder flag. Set it only for networks with Q/DQ layers: without them the
    flag asks for calibration data.
    """
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
        if allow_tf32:
            config.set_flag(trt.BuilderFlag.TF32)
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
        int8_flag = getattr(trt.BuilderFlag, "INT8", None)
        if int8 and int8_flag is not None:
            config.set_flag(int8_flag)
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
            result.update(compare_outputs(logits, reference))
            result["compute_only_images_per_second"] = 1000 * x.shape[0] / result["gpu_compute_ms_mean"]
        if batches is not None:
            print(f"Evaluating {name} on real data (outside timing)", flush=True)

            def predict(images):
                buffers["images"].copy_(images)
                launch()
                return buffers["logits"].cpu()

            result.update(evaluate_batches(predict, batches, x.shape[0]))
        return result


def run_backend(trt, output, name, model, x, reference, options, batches, *, export_model, evaluate_batches,
                compare_outputs, allow_tf32, int8=False):
    """Time batch 1 only; build an untimed accuracy engine when batch sizes differ."""
    callbacks = dict(evaluate_batches=evaluate_batches, compare_outputs=compare_outputs,
                     allow_tf32=allow_tf32, int8=int8)
    shared = batches is not None and options.eval_batch_size == x.shape[0]
    result = benchmark_engine(trt, output, name, x, reference, options, batches if shared else None, **callbacks)
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
        export_model(model, eval_x, output / f"{eval_name}.onnx")
        accuracy = benchmark_engine(trt, output, eval_name, eval_x, eval_reference, options, batches, measure=False, **callbacks)
        engine_fields = {"cuda_stream", "storage_engine_bytes", "storage_onnx_bytes"}
        result.update({key: value for key, value in accuracy.items() if key not in engine_fields})
    result.update(eval_batch_size=options.eval_batch_size, eval_engine_name=eval_name,
                  eval_storage_engine_bytes=accuracy["storage_engine_bytes"],
                  eval_storage_onnx_bytes=accuracy["storage_onnx_bytes"])
    return result


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



def compare_logits(logits, reference):
    """Classification agreement, chosen explicitly by export benchmark recipes."""
    return dict(synthetic_native_vs_reference_top1_agreement=
                (logits.argmax(-1) == reference.argmax(-1)).float().mean().item())


def deployment_latency(model, *, device, image_size, warmup, repeats, output, int8_export=None, sample=None,
                       name="engine"):
    """TensorRT batch-1 latency of the evaluated model, for metrics.json.

    int8_export: callable returning the INT8 Q/DQ export copy (QAT experiments); fields are integer_latency_*.
    Without it the model itself is timed as an FP32 engine (A1); fields are tensorrt_fp32_latency_*.
    Fields stay None with a note when there is no CUDA GPU or TensorRT, or when the engine fails, so the
    rest of an evaluation is never lost. Files go to output/tensorrt/.
    """
    prefix = "integer_latency" if int8_export is not None else "tensorrt_fp32_latency"
    fields = {f"{prefix}_ms": None, f"{prefix}_ms_mean": None, f"{prefix}_ms_p95": None,
              f"{prefix}_backend": None, f"{prefix}_note": None}
    note = f"{prefix}_note"
    if not str(device).startswith("cuda"):
        fields[note] = "TensorRT needs a CUDA GPU: evaluate with --device cuda"
        return fields
    try:
        trt = require_tensorrt(device)
        import onnx  # noqa: F401  (export dependency)
    except (RuntimeError, ImportError) as error:
        fields[note] = f"TensorRT unavailable: {error}"
        return fields
    try:
        net = int8_export() if int8_export is not None else copy.deepcopy(model).cpu().float().eval()
        x = (torch.zeros(1, 3, image_size, image_size) if sample is None else sample).detach().cpu().float()
        with torch.inference_mode():
            reference = net(x)
        directory = Path(output) / "tensorrt"
        directory.mkdir(parents=True, exist_ok=True)
        export_onnx(net, x, directory / f"{name}.onnx")
        options = SimpleNamespace(device=device, warmup=warmup, iterations=repeats)
        result = benchmark_engine(trt, directory, name, x, reference, options, evaluate_batches=evaluate_fixed_batch,
                                  compare_outputs=compare_logits, allow_tf32=False, int8=int8_export is not None)
    except Exception as error:  # Keep the evaluation; report why the engine could not be measured.
        fields[note] = f"TensorRT measurement failed: {type(error).__name__}: {str(error).splitlines()[0]}"
        return fields
    fields.update({
        f"{prefix}_ms": result["gpu_compute_ms_median"], f"{prefix}_ms_mean": result["gpu_compute_ms_mean"],
        f"{prefix}_ms_p95": result["gpu_compute_ms_p95"],
        f"{prefix}_backend": f"tensorrt_{trt.__version__}_{'int8' if int8_export is not None else 'fp32'}",
        f"{prefix}_note": ("batch 1 GPU compute (CUDA events), TF32 off; INT8 Q/DQ products with the trained grids, "
                           "FP32 softmax/norm/GELU/residual; check the layers file for the precision TensorRT chose"
                           if int8_export is not None else "batch 1 GPU compute (CUDA events), FP32 engine, TF32 off"),
        f"{prefix}_engine_bytes": result["storage_engine_bytes"],
        f"{prefix}_layers_file": str(directory / f"{name}.layers.json"),
        f"{prefix}_vs_pytorch_max_abs_logit_diff": result["synthetic_native_vs_reference_max_abs_error"],
    })
    return fields
