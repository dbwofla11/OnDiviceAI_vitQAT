import torch
from torch.utils.data import TensorDataset

from vitqat.cli import parser
from vitqat.runner import run
from vitqat.train import build_pruned_model, measure_model, train_epoch


def test_qat_cls_pruning_train_and_inference_pipeline(tmp_path):
    torch.set_num_threads(2)
    args = parser().parse_args([
        "--mode", "qat", "--cls-pruning", "train_and_inference", "--cls-pruning-rate", "1",
        "--image-size", "64", "--output", str(tmp_path / "out"), "--device", "cpu", "--epochs", "1",
        "--warmup-epochs", "0", "--workers", "0", "--batch-size", "2",
        "--latency-warmup", "0", "--latency-repeats", "1", "--experiment", "cls_test",
    ])

    def data_factory(config):
        splits = [TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])) for _ in range(3)]
        return *splits, {"sha256": "generated-fixture"}

    # Same weight-decay groups as the dense baseline: pos_embed/cls_token excluded under the wrapper.
    model = build_pruned_model(args)
    assert {"backbone.pos_embed", "backbone.cls_token"} <= model.no_weight_decay()

    result = run(args, build_model=build_pruned_model, measure_model=measure_model,
                 train_epoch_fn=train_epoch, data_factory=data_factory)
    assert result["cls_pruning"] == "train_and_inference" and result["token_pruning_ratio"] > 0
