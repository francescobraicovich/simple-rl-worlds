import math
import torch
import torch.nn as nn
from typing import Optional

from .encoder import RotaryEmbedding, rotate_half, DropPath, MultiHeadSelfAttention, MLP

class ModulatedGroupNorm(nn.Module):
    """ Modulated Group Normalization """
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
    
    def forward(self, x, cond):
        scale, shift = cond.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + scale) + shift

class ModulatedLayerNorm(nn.Module):
    """ Modulated Layer Normalization """
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x, cond):
        scale, shift = cond.chunk(2, dim=1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift

class ResidualConvBlock3D(nn.Module):
    """ Residual 3D convolutional block with modulated normalization. """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = ModulatedGroupNorm(channels, num_groups)
        self.act = nn.GELU()

    def forward(self, x, cond):
        out = self.conv(x)
        out = self.norm(out, cond)
        out = self.act(out)
        return x + out

class ModulatedTransformerBlock(nn.Module):
    """ Transformer block with modulated layer normalization. """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.norm1 = ModulatedLayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate
        )
        self.drop_path1 = DropPath(drop_path_rate)
        
        self.norm2 = ModulatedLayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, drop_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x, cond1, cond2):
        x = x + self.drop_path1(self.attn(self.norm1(x, cond1)))
        x = x + self.drop_path2(self.mlp(self.norm2(x, cond2)))
        return x

class UpsampleBlock(nn.Module):
    """
    Single upsampling block with modulated convolution and spatial self-attention.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, drop_path_rate, num_groups=32):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        self.res_conv = ResidualConvBlock3D(embed_dim, num_groups)
        self.transformer = ModulatedTransformerBlock(
            embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, drop_path_rate
        )

    def forward(self, feat, cond_conv, cond_attn1, cond_attn2):
        feat = self.upsample(feat)
        feat = self.conv(feat)
        
        feat = self.res_conv(feat, cond_conv)
        
        B, C, T, H, W = feat.shape
        
        # Apply transformer attention only to temporal patches to avoid memory explosion
        # Process each spatial location independently across time
        feat_for_attn = feat.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)  # [BHW, C, T]
        feat_for_attn = feat_for_attn.transpose(1, 2)  # [BHW, T, C]
        
        # Expand conditioning for all spatial locations
        cond_attn1_expanded = cond_attn1.unsqueeze(1).expand(-1, H * W, -1).reshape(B * H * W, -1)
        cond_attn2_expanded = cond_attn2.unsqueeze(1).expand(-1, H * W, -1).reshape(B * H * W, -1)
        
        feat_attn_out = self.transformer(feat_for_attn, cond_attn1_expanded, cond_attn2_expanded)
        
        # Reshape back to original format
        feat_attn_out = feat_attn_out.transpose(1, 2).reshape(B, H, W, C, T)  # [B, H, W, C, T]
        feat = feat_attn_out.permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]
        
        return feat

class HybridConvTransformerDecoder(nn.Module):
    """
    Generative U-Net style decoder with latent conditioning (AdaLN) and spatial self-attention.
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
        num_upsampling_blocks: Optional[int] = None,
        patch_size_h: int = 8,
        patch_size_w: int = 8,
        num_groups: int = 32
    ):
        super().__init__()
        self.H_p = img_h // patch_size_h
        self.W_p = img_w // patch_size_w
        self.decoder_embed_dim = decoder_embed_dim
        self.embed_dim = embed_dim

        if num_upsampling_blocks is None:
            if img_h % patch_size_h != 0 or img_w % patch_size_w != 0:
                raise ValueError("Image dimensions must be divisible by patch sizes.")
            upsampling_factor = patch_size_h
            if upsampling_factor <= 0 or not math.log2(upsampling_factor).is_integer():
                raise ValueError("Patch size must be a positive power of 2.")
            num_upsampling_blocks = int(math.log2(upsampling_factor))
        
        self.num_upsampling_blocks = num_upsampling_blocks

        cond_size = self.num_upsampling_blocks * self.decoder_embed_dim * 2 * 3
        self.cond_mlp = nn.Sequential(
            nn.Linear(embed_dim, decoder_embed_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_embed_dim * 4, cond_size)
        )

        self.token_decoder = nn.Linear(embed_dim, self.H_p * self.W_p * self.decoder_embed_dim)

        dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, self.num_upsampling_blocks)]
        self.blocks = nn.ModuleList([
            UpsampleBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                drop_rate,
                attn_drop_rate,
                dpr[i],
                num_groups
            ) for i in range(self.num_upsampling_blocks)
        ])

        self.final_norm = nn.GroupNorm(num_groups, decoder_embed_dim)
        self.final_act = nn.GELU()
        self.final_conv = nn.Conv3d(decoder_embed_dim, 1, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        if hasattr(self, 'cond_mlp'):
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, latent_token: torch.Tensor) -> torch.Tensor:
        B, _, E = latent_token.shape
        assert latent_token.shape[1] == 1, f"Expected single token, got {latent_token.shape[1]} tokens"
        
        token = latent_token.squeeze(1)

        cond_params_flat = self.cond_mlp(token)
        
        feat_flat = self.token_decoder(token)
        feat = feat_flat.reshape(B, self.H_p, self.W_p, self.decoder_embed_dim)
        feat = feat.permute(0, 3, 1, 2).unsqueeze(2)

        cond_params_per_block = cond_params_flat.chunk(self.num_upsampling_blocks, dim=1)
        
        for i, block in enumerate(self.blocks):
            cond_conv, cond_attn1, cond_attn2 = cond_params_per_block[i].chunk(3, dim=1)
            feat = block(feat, cond_conv, cond_attn1, cond_attn2)

        feat = self.final_norm(feat)
        feat = self.final_act(feat)
        recon = self.final_conv(feat)
        
        return recon