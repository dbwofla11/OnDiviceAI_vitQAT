"""A4 owns backbone initialization and the model used by training and inference."""
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from utils.constants import NUM_CLASSES
from .quantization import prepare_qat, restore_qat_state
from .pruning import PruningConfig, attach_pruning


def build_backbone(args):
    initialize = bool(getattr(args, "pretrained", False) and not args.resume and not args.eval_checkpoint)
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
    backbone = model
    model = attach_pruning(backbone, config=PruningConfig(pruning_rate=args.cls_pruning_rate),
                           apply_during="train_and_inference")
    model.no_weight_decay = lambda: {f"backbone.{name}" for name in backbone.no_weight_decay()}
    return model


def restore_model(model, checkpoint):
    """Restore only the checkpoint representation supported by this experiment."""
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must include its original model config")
    if (config.get("mode") != 'qat' or config.get("use_cropr") != False
            or config.get("cls_pruning") != 'train_and_inference'):
        raise ValueError("A4 requires mode='qat', use_cropr=False, cls_pruning='train_and_inference'")
    restore_qat_state(model, checkpoint)
