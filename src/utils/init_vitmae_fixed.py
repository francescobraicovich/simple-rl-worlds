#!/usr/bin/env python3
"""
Fixed ViT MAE initialization with better configuration for training stability.
"""

from transformers import ViTMAEConfig, ViTMAEForPreTraining
from .init_models import load_config


def init_vit_mae_fixed(config_path: str = None):
    """
    Initialize the MAE model with improved configuration for training stability.
    
    Key fixes:
    - Larger model (192 hidden size, 4 layers) for better gradient flow
    - Proper MAE mask ratio (0.75)
    - Better numerical stability (layer_norm_eps=1e-6)
    - Dropout for regularization
    - Normalized pixel loss
    - Better initialization
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized MAE model with stable configuration.
    """
    # Load user config for any overrides
    user_config = load_config(config_path)
 
    config = ViTMAEConfig(
            # encoder - increased size for better gradient flow and stability
            hidden_size=192,               # increased from 64 for stability
            num_hidden_layers=4,           # increased from 2 for better representation
            num_attention_heads=8,         # 192/8 = 24-dim per head
            intermediate_size=512,         # 4× hidden_size (was 128)
            hidden_act="gelu",
            hidden_dropout_prob=0.1,       # add dropout for regularization (was 0.0)
            attention_probs_dropout_prob=0.1,  # add attention dropout (was 0.0)
            layer_norm_eps=1e-6,           # better numerical stability (was 1e-12)

            # image & patch sizing
            image_size=84,                  # match your 84×84 frames
            patch_size=7,                   # 84/7 = 12 patches per side
            num_channels=4,                 # stack of 4 grayscale frames

            # decoder - slightly larger for better reconstruction
            decoder_hidden_size=128,       # increased from 64
            decoder_num_hidden_layers=2,   # keep lightweight
            decoder_num_attention_heads=8, # 128/8 = 16-dim per head (was 16 heads)
            decoder_intermediate_size=256, # 2× decoder hidden size

            # masking & loss - critical for MAE training
            mask_ratio=0.75,               # proper MAE masking ratio (was 0.35)
            norm_pix_loss=True,            # normalize pixel loss for stability (was False)

            # better initialization
            initializer_range=0.02         # slightly increased for better init (was 0.01)
        )

    model = ViTMAEForPreTraining(config)
    
    print(f"✅ Initialized ViT MAE with improved configuration:")
    print(f"   - Hidden size: {config.hidden_size} (encoder), {config.decoder_hidden_size} (decoder)")
    print(f"   - Layers: {config.num_hidden_layers} (encoder), {config.decoder_num_hidden_layers} (decoder)")
    print(f"   - Mask ratio: {config.mask_ratio}")
    print(f"   - Normalized pixel loss: {config.norm_pix_loss}")
    print(f"   - Dropout: {config.hidden_dropout_prob}")
    print(f"   - Layer norm eps: {config.layer_norm_eps}")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model
