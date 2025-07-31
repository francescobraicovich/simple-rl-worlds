# encoder.py

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from typing import Optional

from dataclasses import dataclass

@dataclass
class ViTConfig:
    # encoder
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 512
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12

    # image & patch
    image_size: int = 84
    patch_size: int = 7
    num_channels: int = 4

    # decoder (ViT‑MAE)
    decoder_hidden_size: int = 128
    decoder_num_hidden_layers: int = 4
    decoder_num_attention_heads: int = 16
    decoder_intermediate_size: int = 512

    # masking & loss
    mask_ratio: float = 0.35

def _get_activation_fn(act: str) -> nn.Module:
    if act.lower() == "gelu":
        return nn.GELU()
    elif act.lower() == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {act}")


class PatchEmbed(nn.Module):
    """
    Split image into patches and project to hidden dim.
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        img_size, patch_size = config.image_size, config.patch_size
        grid_size = img_size // patch_size
        self.num_patches = grid_size * grid_size

        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)                # [B, D, H/P, W/P]
        x = x.flatten(2)                # [B, D, N]
        x = x.transpose(1, 2)           # [B, N, D]
        return x


class EncoderBlock(nn.Module):
    """
    Single Transformer block.
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            _get_activation_fn(config.hidden_act),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‑attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.drop1(attn_out)

        # MLP
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.mlp(x_norm)
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder that can take a subset of patches (for MAE).
    If `ids_keep` is provided, only those patch embeddings get fed
    through the transformer (with their corresponding pos‑embeddings).
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # patch embedding
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        # cls token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.pos_drop = nn.Dropout(config.hidden_dropout_prob)

        # transformer blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Args:
          x: [B, C, H, W] input images
          ids_keep: optional LongTensor [B, N_vis] with patch indices to keep
                    (for MAE, obtained from your masking function)

        Returns:
          Tensor of shape [B, N_vis+1, D] if ids_keep given,
                        else [B, N+1, D]
        """
        B = x.size(0)
        # 1) Patch embed → [B, N, D]
        x = self.patch_embed(x)
        N, D = x.size(1), x.size(2)

        # 2) If masking: gather only visible patches + their pos‑embeddings
        if ids_keep is not None:
            # gather patch embeddings
            x = torch.gather(
                x, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, N_vis, D]

            # gather corresponding positional embeddings
            pos_patches = self.pos_embed[:, 1:, :].expand(B, N, D)
            pos_patches = torch.gather(
                pos_patches, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, N_vis, D]
        else:
            # use all patches
            pos_patches = self.pos_embed[:, 1:, :].expand(B, N, D)

        # 3) prepend cls token (with its pos‑emb)
        cls_tokens = self.cls_token.expand(B, -1, -1)            # [B,1,D]
        pos_cls = self.pos_embed[:, :1, :].expand(B, -1, -1)     # [B,1,D]

        x = torch.cat(
            (cls_tokens + pos_cls, x + pos_patches),
            dim=1
        )  # [B, N_vis+1, D] or [B, N+1, D]

        # 4) dropout + transformer blocks
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


if __name__ == "__main__":
    enc = VisionTransformerEncoder(ViTConfig())
    dummy_x = torch.randn(8, 4, 84, 84)
    z = enc(dummy_x)
    print("Encoder output shape:", z.shape)  # expect (8, 512)