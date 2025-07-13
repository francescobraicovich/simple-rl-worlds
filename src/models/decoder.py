import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import RotaryEmbedding, rotate_half, DropPath


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with Rotary Position Embeddings.
    Queries from feature map, keys/values from spatial tokens derived from single latent token.
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

        # Rotary embeddings for queries and keys separately
        cos_q, sin_q = self.rotary_emb(Nq, feat.device)
        cos_k, sin_k = self.rotary_emb(Nt, feat.device)
        
        # Apply rotary embeddings separately
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k = (k * cos_k) + (rotate_half(k) * sin_k)

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
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(decoder_embed_dim, decoder_embed_dim, kernel_size=3, padding=1)
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
        feat = self.upsample(feat)
        feat = self.conv(feat)
        feat = self.res_conv(feat)
        feat = self.cross_attn(feat, tokens)
        return feat


class HybridConvTransformerDecoder(nn.Module):
    """
    Hybrid Transformer-Convolutional decoder for grayscale frame reconstruction.
    Accepts a single predicted latent token and reconstructs a complete frame.
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
        num_upsampling_blocks: Optional[int] = None,  # Will be auto-calculated if None
        patch_size_h: int = 8,
        patch_size_w: int = 8
    ):
        super().__init__()
        # Compute patch grid dims
        self.H_p = img_h // patch_size_h
        self.W_p = img_w // patch_size_w
        self.decoder_embed_dim = decoder_embed_dim
        self.embed_dim = embed_dim

        # Auto-calculate number of upsampling blocks if not provided
        if num_upsampling_blocks is None:
            # Calculate how many 2x upsampling steps needed to go from patch grid to full image
            # The upsampling factor is how much we need to scale from patch grid to image
            upsampling_factor_h = img_h // patch_size_h  # This is H_p, but we want img_h // H_p
            upsampling_factor_w = img_w // patch_size_w   # This is W_p, but we want img_w // W_p
            
            # The actual upsampling factor is how much we scale from patch grid to full image
            # Since H_p = img_h // patch_size_h, the upsampling factor is patch_size_h (if img_h/patch_size_h has no remainder)
            # But what we really want is: img_h / H_p = img_h / (img_h // patch_size_h) = patch_size_h (approximately)
            # Actually, let's think about this differently:
            # We start with H_p x W_p patches, and want to reach img_h x img_w pixels
            # So upsampling_factor = img_h / H_p = img_h / (img_h // patch_size_h)
            
            # Validate that image dimensions are divisible by patch sizes
            if img_h % patch_size_h != 0:
                raise ValueError(
                    f"Image height ({img_h}) must be divisible by patch height ({patch_size_h}). "
                    f"Current remainder: {img_h % patch_size_h}. "
                    f"Please adjust img_h to be a multiple of patch_size_h."
                )
            if img_w % patch_size_w != 0:
                raise ValueError(
                    f"Image width ({img_w}) must be divisible by patch width ({patch_size_w}). "
                    f"Current remainder: {img_w % patch_size_w}. "
                    f"Please adjust img_w to be a multiple of patch_size_w."
                )
            
            # Now calculate the correct upsampling factors
            upsampling_factor_h = patch_size_h  # How much we need to scale H_p to reach img_h
            upsampling_factor_w = patch_size_w   # How much we need to scale W_p to reach img_w
            
            # Validate that upsampling factors are equal (square upsampling)
            if upsampling_factor_h != upsampling_factor_w:
                raise ValueError(
                    f"Non-square upsampling not supported. "
                    f"Height upsampling factor: {upsampling_factor_h}, "
                    f"Width upsampling factor: {upsampling_factor_w}. "
                    f"To fix this, ensure that patch_size_h == patch_size_w. "
                    f"Current values: patch_size_h={patch_size_h}, patch_size_w={patch_size_w}"
                )
            
            # Check if upsampling factor is a power of 2
            if upsampling_factor_h <= 0:
                raise ValueError(
                    f"Invalid upsampling factor: {upsampling_factor_h}. "
                    f"Patch sizes must be positive."
                )
            
            # Calculate log2 to find number of upsampling blocks needed
            log2_factor = math.log2(upsampling_factor_h)
            
            if not log2_factor.is_integer():
                raise ValueError(
                    f"Upsampling factor {upsampling_factor_h} is not a power of 2. "
                    f"Each upsampling block doubles the spatial resolution, so the "
                    f"patch size must be a power of 2. "
                    f"Current patch_size_h={patch_size_h}, patch_size_w={patch_size_w}. "
                    f"Valid patch sizes: 1, 2, 4, 8, 16, 32, 64, ..."
                )
            
            num_upsampling_blocks = int(log2_factor)
            
            # Additional validation: ensure we don't have zero upsampling blocks
            if num_upsampling_blocks == 0:
                raise ValueError(
                    f"No upsampling needed: patch size is 1x1. "
                    f"This means we're not really using patches. "
                    f"Consider using larger patch sizes."
                )
        
        self.num_upsampling_blocks = num_upsampling_blocks

        # Token decoder: maps single token to spatial tokens
        self.token_decoder = nn.Linear(embed_dim, self.H_p * self.W_p * embed_dim)

        # Initial projection from spatial tokens to feature map
        self.proj = nn.Linear(embed_dim, decoder_embed_dim)

        # Build upsampling blocks
        dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, self.num_upsampling_blocks)]
        self.blocks = nn.ModuleList([
            UpsampleBlock(
                decoder_embed_dim,
                decoder_num_heads,
                attn_drop_rate,
                drop_rate,
                dpr[i]
            ) for i in range(self.num_upsampling_blocks)
        ])

        # Final conv to 1 channel
        self.final_conv = nn.Conv3d(
            decoder_embed_dim, 1,
            kernel_size=1, stride=1
        )
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Initialize linear layers
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Initialize conv3d layers
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Initialize embeddings
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, latent_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_token: [B, 1, embed_dim] - single predicted latent token
        Returns:
            recon: [B, 1, 1, img_h, img_w] - reconstructed frame
        """
        B, _, E = latent_token.shape
        assert latent_token.shape[1] == 1, f"Expected single token, got {latent_token.shape[1]} tokens"
        
        # Squeeze to [B, E] and decode to spatial tokens
        token = latent_token.squeeze(1)  # [B, E]
        spatial_tokens_flat = self.token_decoder(token)  # [B, H_p * W_p * E]
        
        # Reshape to spatial token grid
        spatial_tokens = spatial_tokens_flat.reshape(B, self.H_p * self.W_p, E)  # [B, H_p * W_p, E]
        
        # Project to decoder dimension - these will be used for cross-attention
        projected_tokens = self.proj(spatial_tokens)  # [B, H_p * W_p, decoder_embed_dim]
        
        # Initialize feature map from projected tokens
        feat = projected_tokens.transpose(1, 2).reshape(B, self.decoder_embed_dim, 1, self.H_p, self.W_p)

        # Upsampling stages - use projected_tokens for cross-attention
        for block in self.blocks:
            feat = block(feat, projected_tokens)

        # Final reconstruction
        recon = self.final_conv(feat)
        return recon
