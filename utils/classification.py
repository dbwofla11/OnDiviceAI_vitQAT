"""Explicit CE/Top-1 classification evaluation, selected by each inference recipe."""
import torch
from torch import nn


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, target in batches:
        images, target = images.to(device), target.to(device)
        output = model(images)
        total_loss += nn.functional.cross_entropy(output, target, reduction="sum").item()
        correct += (output.argmax(dim=1) == target).sum().item()
        count += images.shape[0]
    if count == 0:
        raise ValueError("Empty evaluation dataset")
    return dict(loss=total_loss / count, top1_acc=100.0 * correct / count, samples=count)
