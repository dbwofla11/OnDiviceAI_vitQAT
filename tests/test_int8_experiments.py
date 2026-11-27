"""Every experiment measures integer latency through the shared utils.int8 / utils.tensorrt functions.

The INT8 export of each QAT experiment must compute the same products as its fake-quant model, and each
evaluation must hand the shared TensorRT measurement a working export (A1: its FP32 model). TensorRT itself
needs a CUDA GPU; here it is replaced by a recorder and the graphs run in ONNX Runtime when installed.
"""
import importlib

import pytest
import torch

from utils.int8 import Int8Attention, Int8Conv2d, Int8Linear, audit_int8_products, convert_affine_qat, export_onnx, int8_products

QAT = ("A2", "A4", "B1", "B2")
AFFINE = ("A2", "A4", "B2")
# block Linear (4 per block) + attention QK^T and PV (2 per block) + patch embedding + head (+ Cropr scorers)
GRAPH_PRODUCTS = {"A2": 12 * 6 + 2, "A4": 12 * 6 + 2, "B2": 12 * 6 + 2 + 11, "B1": 12 * 4 + 2 + 11}


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def build(name, *extra):
    config = importlib.import_module(f"{name}.config")
    args = config.parser().parse_args(["--output", "unused", "--image-size", "64", "--drop-path", "0", "--device", "cpu",
                                      "--latency-warmup", "0", "--latency-repeats", "1", *extra])
    for option in ("cropr_pruning_rate", "cls_pruning_rate"):
        if hasattr(args, option):
            setattr(args, option, 1)
    torch.manual_seed(0)
    model = importlib.import_module(f"{name}.model").build_model(args)
    images = torch.randn(4, 3, 64, 64)
    if name != "A1":
        with torch.no_grad():  # Calibrate observers (affine) or initialize the LSQ ranges (B1).
            model.train()(images)
            model.train()(images + 0.5)
        if name != "B1":
            importlib.import_module(f"{name}.quantization").freeze_observers(model, True)
    return model.eval(), args


def int8_export(name, model):
    if name == "B1":
        return importlib.import_module("B1.tensorrt_int8").convert_for_tensorrt(model)
    if name == "B2":
        return importlib.import_module("B2.int8").convert_for_tensorrt(model)
    return convert_affine_qat(model)


def first(output):
    return output[0] if isinstance(output, (list, tuple)) else output


@pytest.mark.parametrize("name", AFFINE)
@pytest.mark.parametrize("activation_bits", ["2", "8"])
def test_affine_layers_and_attention_compute_the_fake_quant_products(name, activation_bits):
    model, _ = build(name, "--activation-bits", activation_bits)
    exported = convert_affine_qat(model) if name != "B2" else int8_export(name, model)
    quantization = importlib.import_module(f"{name}.quantization")
    checked = []

    def hook(layer_name):
        def run(module, inputs, output):
            got = exported.get_submodule(layer_name)(inputs[0])
            proj = module.proj if isinstance(module, quantization.QATAttention) else module
            # A code can flip at a rounding boundary: allow a few output steps, keep the mean tiny.
            step = proj.output_quant.fake_quant.scale.item()
            factor = 4 if isinstance(module, quantization.QATAttention) else 2
            assert (got - output).abs().max() <= factor * step + 1e-4 * output.abs().max(), layer_name
            assert (got - output).abs().mean() <= 1e-2 * output.abs().mean() + 1e-6, layer_name
            checked.append(type(exported.get_submodule(layer_name)))
        return run

    types = (quantization.QATLinear, quantization.QATConv2d, quantization.QATAttention)
    handles = [module.register_forward_hook(hook(layer_name)) for layer_name, module in model.named_modules()
               if isinstance(module, types)]
    with torch.inference_mode():
        model(torch.randn(3, 3, 64, 64))
    for handle in handles:
        handle.remove()
    assert checked.count(Int8Attention) == 12 and checked.count(Int8Conv2d) == 1
    assert checked.count(Int8Linear) == 12 * 4 + 1  # qkv/proj/fc1/fc2 per block and the head


@pytest.mark.parametrize("name", QAT)
@pytest.mark.parametrize("bits", [("2", "8"), ("4", "8")])
def test_whole_int8_model_stays_close_to_fake_quant(name, bits):
    # With 2-bit activations a single code flip, or a tie between Cropr scores, can change which tokens are
    # kept; that is checked layer by layer above instead of on whole-model logits.
    model, _ = build(name, "--weight-bits", bits[0], "--activation-bits", bits[1])
    images = torch.randn(16, 3, 64, 64)
    with torch.inference_mode():
        fake, exported = first(model(images)), int8_export(name, model)(images)
    assert exported.shape == fake.shape and torch.isfinite(exported).all()
    assert (fake.argmax(1) == exported.argmax(1)).float().mean() >= 0.8
    assert (fake - exported).abs().max() < 0.5 * fake.abs().max()


def test_a3_pruned_wrapper_routes_on_the_same_probabilities():
    """A3 attaches CLS-attention pruning at evaluation; its router hooks prob_quant of each attention."""
    from A3.pruning import PruningConfig, attach_pruning

    model, _ = build("A2", "--activation-bits", "8")
    pruned = attach_pruning(model, config=PruningConfig(pruning_rate=1), apply_during="inference")
    exported = convert_affine_qat(pruned)
    images = torch.randn(8, 3, 64, 64)
    with torch.inference_mode():
        fake, got = pruned(images), exported(images)
    assert int8_products(exported) == GRAPH_PRODUCTS["A2"]
    assert (fake.argmax(1) == got.argmax(1)).float().mean() >= 0.8
    assert (fake - got).abs().max() < 0.5 * fake.abs().max()


@pytest.mark.parametrize("name", QAT)
def test_int8_graph_has_int8_products_and_runs_in_onnxruntime(name, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model, _ = build(name, "--activation-bits", "8")
    exported = int8_export(name, model)
    images = torch.randn(8, 3, 64, 64)
    graph = export_onnx(exported, images[:1], tmp_path / "model.onnx")
    assert sum(row["int8"] for row in audit_int8_products(graph)) == GRAPH_PRODUCTS[name]
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    results = torch.cat([torch.from_numpy(session.run(None, {"images": images[i:i + 1].numpy()})[0]) for i in range(8)])
    with torch.inference_mode():
        reference = exported(images)
    # ONNX Runtime fuses the Q/DQ into integer kernels with its own rounding; with barely trained models a
    # near-tie between Cropr scores can still change a kept token, so a few images may flip.
    assert (results - reference).abs().max() < 0.1 * max(1.0, reference.abs().max().item())
    assert (results.argmax(1) == reference.argmax(1)).float().mean() >= 0.75


def recorder(calls, fields):
    def deployment_latency(model, *, device, image_size, warmup, repeats, output, int8_export=None, sample=None,
                           name="engine"):
        calls.append(dict(model=model, device=device, image_size=image_size, output=output, name=name,
                          int8=None if int8_export is None else int8_export()))
        return dict(fields)
    return deployment_latency


@pytest.mark.parametrize("name", ("A1", "A2", "A4", "B2"))
def test_evaluation_hands_the_shared_tensorrt_measurement_a_working_model(name, tmp_path, monkeypatch):
    model, args = build(name, "--activation-bits", "8") if name != "A1" else build(name)
    args.output = str(tmp_path)
    calls = []
    fields = ({"integer_latency_ms": 1.5, "integer_latency_backend": "tensorrt_test_int8"} if name != "A1"
              else {"tensorrt_fp32_latency_ms": 2.5, "tensorrt_fp32_latency_backend": "tensorrt_test_fp32"})
    metrics_module = importlib.import_module(f"{name}.metrics")
    monkeypatch.setattr(metrics_module, "deployment_latency", recorder(calls, fields))
    metrics = metrics_module.measure_model(model, args)
    assert len(calls) == 1 and calls[0]["model"] is model and calls[0]["output"] == str(tmp_path)
    if name == "A1":
        assert calls[0]["int8"] is None and metrics["tensorrt_fp32_latency_ms"] == 2.5
        assert metrics["integer_latency_ms"] is None  # FP32 baseline: no integer engine
        return
    assert metrics["integer_latency_ms"] == 1.5 and metrics["integer_latency_backend"] == "tensorrt_test_int8"
    int8 = calls[0]["int8"]
    assert int8_products(int8) > 0 and not any(hasattr(m, "fake_quant") for m in int8.modules() if m is not int8)
    images = torch.randn(8, 3, 64, 64)
    with torch.inference_mode():
        assert (first(model(images)).argmax(1) == int8(images).argmax(1)).float().mean() >= 0.8


def test_b1_uses_tensorrt_on_cuda_and_onednn_on_cpu(tmp_path, monkeypatch):
    from B1 import metrics as b1_metrics

    model, args = build("B1", "--activation-bits", "8")
    calls = []
    monkeypatch.setattr(b1_metrics, "deployment_latency", recorder(calls, {"integer_latency_ms": 3.0}))
    args.output, args.device = str(tmp_path), "cuda"
    assert b1_metrics.integer_latency(model, args) == {"integer_latency_ms": 3.0}
    assert len(calls) == 1 and int8_products(calls[0]["int8"]) == GRAPH_PRODUCTS["B1"]
    args.device = "cpu"
    cpu = b1_metrics.integer_latency(model, args)
    assert len(calls) == 1  # The CPU path uses B1's oneDNN kernels, not TensorRT.
    assert cpu["integer_latency_backend"] in (None, "onednn_u8s8_int32")
