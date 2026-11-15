"""A3 owns backbone initialization and the upstream model used by paired inference."""
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from utils.constants import NUM_CLASSES
from .quantization import prepare_qat, restore_qat_state


def build_backbone(args):
    initialize = False  # A3 always restores an upstream checkpoint.
    base = create_model(args.model, pretrained=initialize, num_classes=NUM_CLASSES,
                             img_size=args.image_size, global_pool="token", drop_path_rate=args.drop_path)
    if not isinstance(base, VisionTransformer):
        raise ValueError("--model must name a timm VisionTransformer architecture")
    if args.image_size <= 0 or any(args.image_size % size for size in base.patch_embed.patch_size):
        raise ValueError("image_size must be positive and divisible by the model patch size")
    for block in base.blocks:
        block.attn.fused_attn = False
    return base


def build_model(args):
    model = build_backbone(args)
    prepare_qat(model, args.weight_bits, args.activation_bits, scope=args.qat_scope)
    return model


def restore_model(model, checkpoint):
    """Restore only the checkpoint representation supported by this experiment."""
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must include its original model config")
    if (config.get("mode") != 'qat' or config.get("use_cropr") != False
            or config.get("cls_pruning") != None):
        raise ValueError("A3 requires mode='qat', use_cropr=False, cls_pruning=None")
    restore_qat_state(model, checkpoint)
