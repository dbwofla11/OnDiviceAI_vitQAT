"""Small model builders shared by the integer-kernel and TensorRT export tests."""
import torch
from torch import nn

from B1.config import parser
from B1.model import build_model
from B1.modules import project_quantizer_steps


def settings(tmp_path, *extra):
    return parser().parse_args([
        "--image-size", "64", "--cropr-pruning-rate", "1", "--device", "cpu", "--drop-path", "0",
        "--latency-warmup", "0", "--latency-repeats", "2", "--output", str(tmp_path / "run"), *extra])


def trained_model(args, seed=0):
    """Calibrate the activation quantizers and move weights/steps away from their initial values."""
    torch.manual_seed(seed)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images, labels = torch.randn(4, 3, 64, 64), torch.arange(4)
    model.train()
    for _ in range(3):
        outputs = model(images)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        loss = sum(nn.functional.cross_entropy(out, labels) for out in outputs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        project_quantizer_steps(model)  # As in train_epoch: without it, lr 1e-3 collapses many 8-bit steps.
    return model.eval()


# Names of the option sets the integer and TensorRT export paths must support.
CONFIGS = {
    "W2A2": [],
    "W2A8": ["--activation-bits", "8"],
    "W4A8": ["--weight-bits", "4", "--activation-bits", "8"],
    "W8A8": ["--weight-bits", "8", "--activation-bits", "8"],
    "W2A2 symmetric activation": ["--activation-quantizer", "lsq"],
    "W2A8 symmetric activation": ["--activation-bits", "8", "--activation-quantizer", "lsq"],
    "W2A2 lsq weights": ["--weight-quantizer", "lsq"],
    "FP32 edges and scorer": ["--first-last-bits", "32", "--cropr-scorer-bits", "32"],
    "2-bit edges and scorer": ["--first-last-bits", "2", "--cropr-scorer-bits", "2"],
    "LLF": ["--cropr-llf"],
    "dense": ["--no-use-cropr"],
}
