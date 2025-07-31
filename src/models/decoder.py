# decoder.py

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
    """Returns activation module given string identifier."""
    act = act.lower()
    if act == "gelu":
        return nn.GELU()
    elif act == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {act}")


class DecoderBlock(nn.Module):
    """
    Single Transformer block for MAE decoder.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        attention_dropout_prob: float,
        layer_norm_eps: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout_prob,
            batch_first=True
        )
        self.drop1 = nn.Dropout(hidden_dropout_prob)

        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, intermediate_size),
            _get_activation_fn(hidden_act),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, embed_dim),
            nn.Dropout(hidden_dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‐attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.drop1(attn_out)

        # MLP
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.mlp(x_norm)
        return x


class Decoder(nn.Module):
    """
    Lightweight MAE decoder head.
    Takes encoder‐visible patch embeddings + mask tokens, restores full sequence,
    applies transformer blocks, and reconstructs pixel patches.
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # number of patches
        patches_per_side = config.image_size // config.patch_size
        self.num_patches = patches_per_side * patches_per_side

        # project encoder dim -> decoder dim
        self.decoder_embed = nn.Linear(
            config.hidden_size,
            config.decoder_hidden_size
        )

        # learnable mask token
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, config.decoder_hidden_size)
        )

        # positional embeddings for decoder (no cls token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.decoder_hidden_size)
        )
        self.pos_drop = nn.Dropout(config.hidden_dropout_prob)

        # transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=config.decoder_hidden_size,
                num_heads=config.decoder_num_attention_heads,
                intermediate_size=config.decoder_intermediate_size,
                hidden_act=config.hidden_act,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_dropout_prob=config.attention_probs_dropout_prob,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.decoder_num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)

        # reconstruction head: from decoder embeddings -> patch pixels
        self.head = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size * config.patch_size * config.num_channels
        )

        self._init_weights()

    def _init_weights(self):
        # initialize positional embeddings and mask token
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

        # initialize decoder_embed & head
        trunc_normal_(self.decoder_embed.weight, std=0.02)
        nn.init.zeros_(self.decoder_embed.bias)
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # init transformer blocks
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.LongTensor
    ) -> torch.Tensor:
        """
        Args:
          x: visible patch embeddings from encoder of shape (B, N_vis, hidden_size)
          ids_restore: indices to restore original patch order,
                       shape (B, N) where N = total patches

        Returns:
          Reconstructed patches of shape (B, N, P*P*C)
        """
        B, N_vis, _ = x.shape

        # 1) embed to decoder dimension
        x = self.decoder_embed(x)  # (B, N_vis, D_dec)

        # 2) prepare mask tokens
        mask_tokens = self.mask_token.expand(B, self.num_patches - N_vis, -1)  # (B, N_mask, D_dec)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N_vis+N_mask, D_dec == N)

        # 3) unshuffle to original patch order
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1))
        )  # (B, N, D_dec)

        # 4) add positional embeddings & dropout
        x_full = x_full + self.pos_embed
        x_full = self.pos_drop(x_full)

        # 5) apply decoder transformer blocks
        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)

        # 6) project to pixel values
        x_rec = self.head(x_full)  # (B, N, patch_dim)

        return x_rec



if __name__ == "__main__":

    dec = Decoder(ViTConfig())
    dummy_z = torch.randn(8, 512)
    out = dec(dummy_z)
    print("Decoder output shape:", out.shape)  # expect (8, 4, 84, 84)
