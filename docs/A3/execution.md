# A3 evaluation

Evaluate a QAT-only checkpoint with and without CLS attention Top-K pruning on the same data. No query checkpoint is needed.

```bash
python -m A3.eval_pruning --checkpoint A3/checkpoints/qat-only/seed42/best.pt --output A3/runs/seed42/eval-pair --device cuda
```

The default `cropr-r16-first17-llf` specifies the Cropr removal schedule: remove 17 patches at the first stage and 16 at each following stage, restoring removed tokens before the final block. Selection uses the CLS-to-patch self-attention probabilities averaged across heads. QAT uses fake-quantized attention probabilities. No learned Cropr query scorer is used.

The evaluator validates the inherited configuration and split hash and writes configuration, split, dense and pruned metrics, and comparison files to a new empty output directory. Both conditions use the same test split and latency sample. Backbone and quantizer state are checked for changes. Fake-quant latency is not integer-kernel latency.

Both conditions restore A2's affine (asymmetric) quantizers unchanged, so inference uses the same quantization grid as A2 training. There is no separate export or symmetric INT8 path.
