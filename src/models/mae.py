from transformers import ViTMAEConfig, ViTMAEModel
import torch

# 1) Define a config that mirrors your CNN‐AE’s latent size (512)…
# 2) …but stitches to 4‐channel, 84×84 inputs via 7×7 patches (12×12 tokens).
config = ViTMAEConfig(
    # encoder (ViT‐Base defaults would be hidden_size=768)
    hidden_size=96,                # your “latent_dim”
    num_hidden_layers=12,           # same depth as ViT‐Base
    num_attention_heads=8,          # 512/8 = 64‑dim per head
    intermediate_size=384,         # 4× hidden_size
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    layer_norm_eps=1e-12,

    # image & patch sizing
    image_size=84,                  # match your 84×84 frames
    patch_size=7,                   # 84/7 = 12 patches per side
    num_channels=4,                 # stack of 4 grayscale frames

    # decoder (lightweight MAE decoder)
    decoder_hidden_size=96,        # mirror encoder latent
    decoder_num_hidden_layers=8,    # shallow “head”
    decoder_num_attention_heads=16, # head_dim = 512/16 = 32
    decoder_intermediate_size=384,

    # masking & loss
    mask_ratio=0.75,                # same 75% masking
    norm_pix_loss=False             # or True if you want pixel‐norm loss
)


# Instantiate the model for masked‐image modeling:
model = ViTMAEModel(config)


if __name__ == "__main__":
    # Print the model architecture
    print(model)

    print('num parameters:', model.num_parameters())

    sample_image = {
        "pixel_values": torch.zeros((8, 4, 84, 84), dtype=torch.float32)
    }
    # Forward pass through the model
    outputs = model(**sample_image)
    print('outputs last_hidden_state shape:', outputs.last_hidden_state.shape)