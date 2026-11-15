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
