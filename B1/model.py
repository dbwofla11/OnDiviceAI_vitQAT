"""B1 owns backbone initialization, Linear QAT conversion and checkpoint restoration."""
import torch
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from utils.constants import NUM_CLASSES
from .cropr_adapter import adapt_model
from .modules import prepare_linear_qat, validate_linear_qat_state


def build_backbone(args):
    initialize = bool(getattr(args, "pretrained", False) and not args.resume and not args.eval_checkpoint)
    base = create_model(args.model, pretrained=initialize, num_classes=NUM_CLASSES,
                             img_size=args.image_size, global_pool="token", drop_path_rate=args.drop_path)
    if not isinstance(base, VisionTransformer):
        raise ValueError("--model must name a timm VisionTransformer architecture")
    if args.image_size <= 0 or any(args.image_size % size for size in base.patch_embed.patch_size):
        raise ValueError("image_size must be positive and divisible by the model patch size")
    # Same construction and RNG use as B2, so a seed gives the same initial FP32 weights.
    return adapt_model(base, use_cropr=args.use_cropr, pruning_rate=args.cropr_pruning_rate,
                       llf=args.cropr_llf, image_size=args.image_size, drop_path=args.drop_path)


def load_fp32_weights(model, path):
    """Initialize from an FP32 run of the same architecture; Cropr modules may be absent."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)  # runner checkpoint dict or a raw model_state.pt
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("cropr.") for key in result.missing_keys):
        raise ValueError(f"--init-checkpoint must be an FP32 model of the same architecture: {result}")


def build_model(args):
    model = build_backbone(args)
    # Resume and evaluation restore the trained state, so only a new run reads the FP32 weights.
    if getattr(args, "init_checkpoint", None) and not args.resume and not args.eval_checkpoint:
        load_fp32_weights(model, args.init_checkpoint)
    # Weight step sizes are initialized from the (loaded) weights, as LSQ prescribes.
    return prepare_linear_qat(model, args.weight_bits, args.activation_bits, args.first_last_bits,
                              args.cropr_scorer_bits, args.weight_quantizer, args.activation_quantizer)


def restore_model(model, checkpoint):
    """Restore only the checkpoint representation supported by this experiment."""
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must include its original model config")
    if config.get("mode") != 'linear_qat' or config.get("cls_pruning") != None:
        raise ValueError("B1 requires mode='linear_qat', cls_pruning=None")
    model.load_state_dict(checkpoint["model"], strict=True)
    validate_linear_qat_state(model)
