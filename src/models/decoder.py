import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RotaryEmbedding, apply_rotary_pos_emb, DropPath


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with Rotary Position Embeddings.
    Queries from feature map, keys/values from latent tokens.
    """
    def __init__(self, embed_dim_q, embed_dim_kv, num_heads,
                 attn_drop_rate=0., proj_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        assert embed_dim_q % num_heads == 0 and embed_dim_kv % num_heads == 0, \
            "embed dims must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim_q = embed_dim_q // num_heads
        self.head_dim_kv = embed_dim_kv // num_heads
        self.scale = self.head_dim_q ** -0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim_q, embed_dim_q, bias=False)
        self.kv_proj = nn.Linear(embed_dim_kv, embed_dim_q * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim_q, embed_dim_q)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.drop_path = DropPath(drop_path_rate)

        # Rotary embeddings for queries and keys
        self.rotary_emb = RotaryEmbedding(self.head_dim_q)

    def forward(self, feat, tokens):
        # feat: [B, Dq, 1, Hp, Wp]
        # tokens: [B, Nt, Dkv]
        B, Dq, T, Hp, Wp = feat.shape
        # Flatten spatial dims
        Nq = T * Hp * Wp
        q = feat.flatten(2).transpose(1, 2)  # [B, Nq, Dq]
        Nt, Dkv = tokens.shape[1], tokens.shape[2]

        # Linear projections
        q = self.q_proj(q)  # [B, Nq, Dq]
        kv = self.kv_proj(tokens)  # [B, Nt, 2*Dq]
        k, v = kv.chunk(2, dim=-1)  # each [B, Nt, Dq]

        # Reshape for heads
        q = q.reshape(B, Nq, self.num_heads, self.head_dim_q).permute(0, 2, 1, 3)  # [B, h, Nq, dh]
        k = k.reshape(B, Nt, self.num_heads, self.head_dim_q).permute(0, 2, 1, 3)      # [B, h, Nt, dh]
        v = v.reshape(B, Nt, self.num_heads, self.head_dim_q).permute(0, 2, 1, 3)      # [B, h, Nt, dh]

        # Rotary embeddings
        cos, sin = self.rotary_emb(Nq, feat.device)
        # Expand cos/sin to match heads: [1,1,Nq,dh]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, Dq)  # [B, Nq, Dq]
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # Residual connection with drop path
        out = q.new_zeros((B, Nq, Dq)).to(out) + out  # ensure same device/dtype
        out = self.drop_path(out)

        # Unflatten back to [B, Dq, T, Hp, Wp]
        out = out.transpose(1, 2).reshape(B, Dq, T, Hp, Wp)
        return feat + out


class ResidualConvBlock3D(nn.Module):
    """
    Residual 3D convolutional block (Conv3D -> Norm -> GELU).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, T, H, W]
        out = self.conv(x)
        # Move channel dim to last for LayerNorm
        B, C, T, H, W = out.shape
        out = out.permute(0, 2, 3, 4, 1).reshape(-1, C)
        out = self.norm(out)
        out = self.act(out)
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return x + out


class UpsampleBlock(nn.Module):
    """
    Single upsampling block: transpose conv -> residual conv -> cross-attention infusion.
    """
    def __init__(self, decoder_embed_dim, num_heads,
                 attn_drop_rate, proj_drop_rate, drop_path_rate):
        super().__init__()
        # Double spatial resolution
        self.upconv = nn.ConvTranspose3d(
            decoder_embed_dim, decoder_embed_dim,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        # Residual conv block
        self.res_conv = ResidualConvBlock3D(decoder_embed_dim)
        # Cross-attention infusion
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim_q=decoder_embed_dim,
            embed_dim_kv=decoder_embed_dim,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            drop_path_rate=drop_path_rate
        )

    def forward(self, feat, tokens):
        feat = self.upconv(feat)
        feat = self.res_conv(feat)
        feat = self.cross_attn(feat, tokens)
        return feat


class HybridConvTransformerDecoder(nn.Module):
    """
    Hybrid Transformer-Convolutional decoder for grayscale frame reconstruction.
    """
    def __init__(
        self,
        img_h: int = 64,
        img_w: int = 64,
        frames_per_clip: int = 1,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        decoder_embed_dim: int = 512,
        decoder_num_layers: int = 3,
        decoder_num_heads: int = 8,
        decoder_drop_path_rate: float = 0.1,
        num_upsampling_blocks: int = 3,
        patch_size_h: int = 8,
        patch_size_w: int = 8
    ):
        super().__init__()
        # Compute patch grid dims
        self.H_p = img_h // patch_size_h
        self.W_p = img_w // patch_size_w
        self.decoder_embed_dim = decoder_embed_dim

        # Initial projection from tokens to feature map
        self.proj = nn.Linear(embed_dim, decoder_embed_dim)

        # Build upsampling blocks
        dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, num_upsampling_blocks)]
        self.blocks = nn.ModuleList([
            UpsampleBlock(
                decoder_embed_dim,
                decoder_num_heads,
                attn_drop_rate,
                drop_rate,
                dpr[i]
            ) for i in range(num_upsampling_blocks)
        ])

        # Final conv to 1 channel
        self.final_conv = nn.Conv3d(
            decoder_embed_dim, 1,
            kernel_size=1, stride=1
        )

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tokens: [B, N_tokens, embed_dim]
        Returns:
            recon: [B, 1, 1, img_h, img_w]
        """
        B, N, E = x_tokens.shape
        # Initial projection and reshape
        x = self.proj(x_tokens)  # [B, N, D]
        feat = x.transpose(1, 2).reshape(B, self.decoder_embed_dim, 1, self.H_p, self.W_p)

        # Upsampling stages
        for block in self.blocks:
            feat = block(feat, x_tokens)

        # Final reconstruction
        recon = self.final_conv(feat)
        return recon
