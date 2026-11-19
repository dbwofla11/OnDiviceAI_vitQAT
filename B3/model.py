"""Construct Cropr from its saved architecture and restore FP32 or PTQ state."""
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from utils.constants import NUM_CLASSES
from .cropr_adapter import adapt_model
from .config import validate_model_config
from .quantization import prepare_ptq, validate_observers


def build_backbone(args):
    initialize = bool(getattr(args, "pretrained", False) and not args.resume and not args.eval_checkpoint)
    base = create_model(args.model, pretrained=initialize, num_classes=NUM_CLASSES,
                        img_size=args.image_size, global_pool="token", drop_path_rate=args.drop_path)
    if not isinstance(base, VisionTransformer):
        raise ValueError("--model must name a timm VisionTransformer architecture")
    if args.image_size <= 0 or any(args.image_size % size for size in base.patch_embed.patch_size):
        raise ValueError("image_size must be positive and divisible by the model patch size")
    return adapt_model(base, use_cropr=True, pruning_rate=args.cropr_pruning_rate,
                       llf=args.cropr_llf, image_size=args.image_size, drop_path=args.drop_path)


def build_model(args):
    model = build_backbone(args)
    if args.mode == "ptq":
        prepare_ptq(model, args.weight_bits, args.activation_bits, args.ptq_scope)
    return model


def restore_model(model, checkpoint):
    config = checkpoint["config"]
    validate_model_config(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if config["mode"] == "ptq":
        validate_observers(model)
