"""Build the EVA02 Food-101 backbone with optional timm pretrained weights."""

from timm import create_model

from .constants import NUM_CLASSES


EVA02_MODEL = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"


def load_eva02_backbone(*, pretrained=True, image_size=448, drop_path_rate=0.0):
    """Keep EVA02's native pooling and initialize a new 101-class classifier."""
    if image_size <= 0 or image_size % 14:
        raise ValueError("EVA02 image_size must be positive and divisible by 14")
    return create_model(EVA02_MODEL, pretrained=pretrained, num_classes=NUM_CLASSES,
                        img_size=image_size, drop_path_rate=drop_path_rate)
