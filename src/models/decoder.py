import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

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


class ResidualConvBlock2D(nn.Module):
    """
    Residual 2D convolutional block (Conv2D -> Norm -> GELU).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.conv(x)
        # Move channel dim to last for LayerNorm
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(-1, C)
        out = self.norm(out)
        out = self.act(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class UpsampleBlock(nn.Module):
    """
    Single upsampling block: transpose conv -> residual conv -> cross-attention infusion.
    """
    def __init__(self, decoder_embed_dim, num_heads,
                 attn_drop_rate, proj_drop_rate, drop_path_rate):
        super().__init__()
        # Double spatial resolution (2D upsampling since temporal dim is always 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=3, padding=1)
        # Residual conv block
        self.res_conv = ResidualConvBlock2D(decoder_embed_dim)
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
        # feat: [B, C, T, H, W] -> squeeze temporal dim since it's always 1
        B, C, T, H, W = feat.shape
        assert T == 1, f"Expected temporal dimension to be 1, got {T}"
        feat_2d = feat.squeeze(2)  # [B, C, H, W]
        
        feat_2d = self.upsample(feat_2d)
        feat_2d = self.conv(feat_2d)
        feat_2d = self.res_conv(feat_2d)
        
        # Add temporal dimension back for cross-attention
        feat_3d = feat_2d.unsqueeze(2)  # [B, C, 1, H, W]
        feat_3d = self.cross_attn(feat_3d, tokens)
        return feat_3d


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

        # Final conv to 1 channel (2D since temporal dimension is always 1)
        self.final_conv = nn.Conv2d(
            decoder_embed_dim, 1,
            kernel_size=1, stride=1
        )

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

        # Final reconstruction - convert to 2D for final conv, then back to 3D
        B, C, T, H, W = feat.shape
        assert T == 1, f"Expected temporal dimension to be 1, got {T}"
        feat_2d = feat.squeeze(2)  # [B, C, H, W]
        recon_2d = self.final_conv(feat_2d)  # [B, 1, H, W]
        recon = recon_2d.unsqueeze(2)  # [B, 1, 1, H, W]
        return recon


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for reconstructing RGB images from latent vectors.
    
    Architecture:
    - Input: [B, 1, embed_dim] latent vector from predictor
    - Linear layer: latent_dim → initial_channels × initial_size × initial_size
    - Multiple ConvTranspose2d layers with SiLU + LayerNorm
    - Output: [B, output_channels, 1, height, width] reconstructed frame
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        initial_size: int = 4,
        conv_channels: list = None,
        activation: str = 'silu',
        dropout_rate: float = 0.0,
        target_height: int = 224,
        target_width: int = 224
    ):
        super().__init__()
        
        # Set default conv_channels if not provided
        if conv_channels is None:
            conv_channels = [256, 128, 64, 32, 3]  # Changed last channel to 3 for RGB
        
        self.config = {
            'latent_dim': latent_dim,
            'initial_size': initial_size,
            'conv_channels': conv_channels,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'target_height': target_height,
            'target_width': target_width
        }
        
        # Extract configuration
        self.latent_dim = self.config['latent_dim']
        initial_size = self.config['initial_size']
        conv_channels = self.config['conv_channels']
        target_height = self.config['target_height']
        target_width = self.config['target_width']
        
        # Get activation function
        if self.config['activation'].lower() == 'silu':
            activation_fn = nn.SiLU
        elif self.config['activation'].lower() == 'relu':
            activation_fn = nn.ReLU
        elif self.config['activation'].lower() == 'gelu':
            activation_fn = nn.GELU
        else:
            activation_fn = nn.SiLU  # Default fallback
        
        # Initial linear layer: latent_dim → initial_channels × initial_size × initial_size
        self.initial_linear = nn.Linear(
            self.latent_dim, 
            conv_channels[0] * initial_size * initial_size
        )
        
        # Store initial spatial size and first channel count
        self.initial_size = initial_size
        self.initial_channels = conv_channels[0]
        
        # Build transpose convolution layers with LayerNorm
        self.conv_transpose_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(len(conv_channels) - 1):
            in_channels = conv_channels[i]
            out_channels = conv_channels[i + 1]
            
            # Transpose convolution layer
            conv_transpose = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            )
            self.conv_transpose_layers.append(conv_transpose)
            
            # LayerNorm for all layers except the final one
            if i < len(conv_channels) - 2:  # Not the final layer
                self.layer_norms.append(nn.LayerNorm(out_channels))
            else:
                self.layer_norms.append(None)  # No norm for final layer
        
        # Activation function
        self.activation = activation_fn()
        
        # Dropout layer
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, latent_token: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            latent_token: Input tensor of shape [B, 1, embed_dim]
            
        Returns:
            Reconstructed image of shape [B, output_channels, 1, height, width]
        """
        # Input shape: [B, 1, embed_dim]
        B, T, E = latent_token.shape
        assert T == 1, f"Expected single token, got {T} tokens"
        assert E == self.latent_dim, f"Expected embed_dim {self.latent_dim}, got {E}"
        
        # Squeeze token dimension: [B, 1, embed_dim] → [B, embed_dim]
        x = latent_token.squeeze(1)
        
        # Linear layer: [B, embed_dim] → [B, initial_channels*initial_size*initial_size]
        x = self.initial_linear(x)
        
        # Reshape to feature map: [B, initial_channels*initial_size*initial_size] → [B, initial_channels, initial_size, initial_size]
        x = x.reshape(B, self.initial_channels, self.initial_size, self.initial_size)
        
        # Pass through transpose convolution layers
        for i, (conv_transpose, layer_norm) in enumerate(zip(self.conv_transpose_layers, self.layer_norms)):
            # Apply transpose convolution
            x = conv_transpose(x)
            
            # Apply activation (for all layers)
            if i < len(self.conv_transpose_layers) - 1:  # Not the final layer
                x = self.activation(x)
                
                # Apply LayerNorm if available
                if layer_norm is not None:
                    # Reshape for LayerNorm: [B, C, H, W] → [B, H, W, C]
                    x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
                    x = layer_norm(x)          # Apply LayerNorm on channel dimension
                    x = x.permute(0, 3, 1, 2)  # Back to [B, C, H, W]
                    
                    # Apply dropout
                    x = x.permute(0, 2, 3, 1)  # [B, H, W, C] for dropout
                    x = self.dropout(x)
                    x = x.permute(0, 3, 1, 2)  # Back to [B, C, H, W]
        
        # Add temporal dimension: [B, C, H, W] → [B, C, 1, H, W]
        output = x.unsqueeze(2)
        
        # Ensure output matches target size exactly
        B, C, T, H, W = output.shape
        if H != self.config['target_height'] or W != self.config['target_width']:
            output = F.interpolate(
                output.squeeze(2),  # Remove temporal dim for interpolation
                size=(self.config['target_height'], self.config['target_width']),
                mode='bilinear',
                align_corners=False
            ).unsqueeze(2)  # Add temporal dim back
        
        return output
