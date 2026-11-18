"""Load a timm backbone with caller-selected pretrained weights."""

from pathlib import Path

from timm import create_model

from .constants import NUM_CLASSES


def load_pretrained_backbone(
    model_name: str,
    *,
    weights: str | None = None,
    weights_path: str | Path | None = None,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    **model_kwargs,
):
    """Select a timm weights tag or load a compatible local state-dict file."""
    if not pretrained and (weights is not None or weights_path is not None):
        raise ValueError("Set pretrained=True when specifying pretrained weights")

    if weights_path is not None:
        # Use timm's pretrained path so it can adapt the classification head.
        # The file may be a state dict even if the registry tag uses a custom loader.
        model_kwargs["pretrained_cfg_overlay"] = {
            **(model_kwargs.get("pretrained_cfg_overlay") or {}),
            "source": "",
            "state_dict": None,
            "file": str(weights_path),
            "custom_load": False,
        }

    return create_model(model_name, pretrained=pretrained, pretrained_cfg=weights,
                        num_classes=num_classes, **model_kwargs)
