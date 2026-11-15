"""Construct the unmodified Cropr model from a timm backbone, preserving its weights."""
import cropr as cropr_package
from cropr.cls.cropr import Cropr, CrossAttention

# The upstream script uses `from cropr import Cropr`. Expose the same class when
# imported as a package, without editing upstream files or changing sys.path.
cropr_package.Cropr = Cropr
from cropr.cls.vision_transformer import VisionTransformer as CroprViT

def adapt_model(base, *, use_cropr=False, pruning_rate=8, llf=False, image_size=224, drop_path=0.1):
    if base.num_prefix_tokens != 1:
        raise ValueError("The original Cropr model requires one CLS prefix token")
    if use_cropr:
        pruners = len(base.blocks) - (2 if llf else 1)
        if pruners < 1 or pruning_rate < 0:
            raise ValueError("Cropr needs at least one pruning block and a nonnegative rate")
        if pruners * pruning_rate + 1 >= base.patch_embed.num_patches + 1:
            raise ValueError("Cropr schedule would remove all tokens; reduce pruning rate")
    # The original constructor fixes its transformer blocks to this standard ViT recipe.
    for block in base.blocks:
        if (block.mlp.fc1.out_features != 4 * base.embed_dim or block.attn.qkv.bias is None
                or block.norm1.eps != 1e-6 or block.norm2.eps != 1e-6):
            raise ValueError("Architecture is incompatible with the original Cropr transformer blocks")
    cfg = dict(use_cropr=use_cropr, pruning_rate=pruning_rate, llf=llf,
               num_queries=1, num_heads=1, pre_attn_norm=False, q_proj=False, k_proj=False,
               v_proj=False, mlp=True, mlp_ratio=4, training=True)
    model = CroprViT(cfg, num_classes=base.num_classes, img_size=image_size,
                     patch_size=base.patch_embed.patch_size[0], embed_dim=base.embed_dim,
                     depth=len(base.blocks), num_heads=base.blocks[0].attn.num_heads,
                     mlp_ratio=4, global_pool="token", drop_path_rate=drop_path)
    # Keep the same backbone initialization whether Cropr is on or off.
    result = model.load_state_dict(base.state_dict(), strict=False)
    if result.unexpected_keys or any(not key.startswith("cropr.") for key in result.missing_keys):
        raise ValueError(f"Architecture is incompatible with the original Cropr: {result}")
    for block in model.blocks:
        block.attn.fused_attn = False
    return model
