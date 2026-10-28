# A3 evaluation

Evaluate a QAT-only checkpoint with and without CLS attention Top-K pruning on the same data. No query checkpoint is needed.

```bash
python -m A3.eval_pruning --checkpoint A3/checkpoints/qat-only/seed42/best.pt --output A3/runs/seed42/eval-pair --device cuda
```

The default `cropr-r16-first17-llf` specifies the Cropr removal schedule: remove 17 patches at the first stage and 16 at each following stage, restoring removed tokens before the final block. Selection uses the CLS-to-patch self-attention probabilities averaged across heads. QAT uses fake-quantized attention probabilities. No learned Cropr query scorer is used.

The evaluator validates the inherited configuration and split hash and writes configuration, split, dense and pruned metrics, and comparison files to a new empty output directory. Both conditions use the same test split and latency sample. Backbone and quantizer state are checked for changes. Fake-quant latency is not integer-kernel latency.

## A2 checkpoint + A3 pruning + TensorRT INT8

Use a calibrated **W2A8 QAT-only A2 checkpoint**. The runner builds both dense and
CLS-pruned engines from the same checkpoint, preserving the R=16/first17/LLF schedule.
No A3 training or learned query checkpoint is needed. Other weight/activation bit
settings are rejected by the current converter.

In an environment with CUDA-enabled PyTorch and an NVIDIA GPU:

```powershell
python -m pip install -r requirements-tensorrt.txt
python -m A3.benchmark_tensorrt_int8 `
  --checkpoint "A2/runs/seed42/best.pt" `
  --output "A3/runs/seed42/tensorrt-int8" `
  --data-root "data" `
  --eval-split test --eval-batch-size 32 `
  --warmup 20 --iterations 100
```

Replace the example checkpoint and data paths with actual locations. If omitted,
`--data-root` inherits the checkpoint setting; data is never downloaded automatically.
Use a new empty output directory. Reduce `--eval-batch-size` if GPU memory is limited.
Native execution always evaluates the full selected split and verifies its hash.

CPU export smoke check (no TensorRT or dataset needed):

```powershell
python -m A3.benchmark_tensorrt_int8 `
  --checkpoint "A2/runs/seed42/best.pt" `
  --output "A3/runs/seed42/int8-export" --export-only
```

Outputs include `dense_int8.onnx`, `a3_int8.onnx`, `report.json`, and `metrics.csv`.
Native execution additionally writes engines, layer inspection reports, raw timings,
and split metadata. Accuracy engines use the requested evaluation batch; timing uses
batch 1, the same first image of the selected split, and excludes engine building,
data loading, and transfers. The report compares INT8 dense/pruned Top-1 and median
latency, and records each corresponding source fake-quant accuracy. GFLOPs and token
counts describe the source architecture, not measured TensorRT kernel arithmetic.

W2 values are stored in INT8 Q/DQ, not executed with INT2 kernels. Activation ranges
are remapped to symmetric INT8 without new calibration, so accuracy and selected
tokens may change. Equal attention scores may also sort differently across runtimes.
Inspect `*.layers.json` to establish actual INT8 kernel use; export success alone
does not establish acceleration. `--export-only` does not measure real accuracy or
native latency. CPU ONNX tests and CUDA-gated native tests live in
`A3/tests/test_tensorrt_int8.py`.

TensorRT runs on a dedicated non-default CUDA stream, shared by input copies,
enqueue, timing events, and output copies. Blocking CPU output copies complete
before accuracy is computed. This avoids the default-stream synchronization warning;
rerun both conditions after updating when comparing latency with older results.
A3 shape validation remains active in eager execution but is omitted during
fixed-shape ONNX tracing. Other library-level tracing warnings may still appear.
