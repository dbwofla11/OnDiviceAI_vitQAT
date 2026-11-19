"""Predict Food-101 classes for one image using an FP32 or PTQ checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.constants import NUM_CLASSES
from utils.reproducibility import configure_backend
from .config import CHECKPOINT_DEFAULTS, validate_model_config
from .model import build_model, restore_model


def food101_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize(round(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def class_names(checkpoint, directory):
    if "classes" in checkpoint:
        return checkpoint["classes"]
    labels = directory / "classes.json"
    if labels.is_file():
        return json.loads(labels.read_text(encoding="utf-8"))
    return json.loads((directory / "split.json").read_text(encoding="utf-8"))["classes"]


@torch.inference_mode()
def predict(checkpoint_path, image_path, device, topk):
    configure_backend(deterministic=True, allow_tf32=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = {**CHECKPOINT_DEFAULTS, **checkpoint["config"]}
    validate_model_config(config)
    config.update(resume=None, eval_checkpoint=str(checkpoint_path))
    model = build_model(argparse.Namespace(**config))
    restore_model(model, checkpoint)
    model.to(device).eval()
    classes = class_names(checkpoint, Path(checkpoint_path).parent)
    if len(classes) != NUM_CLASSES or not 1 <= topk <= NUM_CLASSES:
        raise ValueError("Expected 101 Food-101 class names and topk between 1 and 101")
    with Image.open(image_path) as image:
        x = food101_eval_transform(config["image_size"])(image.convert("RGB"))
    scores, indices = model(x.unsqueeze(0).to(device))[0].softmax(0).topk(topk)
    return [{"class_id": i, "class": classes[i], "probability": p}
            for i, p in zip(indices.tolist(), scores.tolist())]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--topk", type=int, default=5)
    opts = p.parse_args()
    if opts.threads > 0:
        torch.set_num_threads(opts.threads)
    print(json.dumps(predict(opts.checkpoint, opts.image, opts.device, opts.topk), indent=2))
