import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for self-attention.
    Generates cos and sin embedding tables for a given sequence length.
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        # Create position ids
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        # Outer product to get frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Duplicate for sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary embeddings to queries and keys.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FactorizedPatchEmbed(nn.Module):
    """
    Spatial patch embedding for grayscale video.
    Applies a spatial convolution to each frame independently.
    (Corrected Version)
    """
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        # This layer is a 3D conv with a kernel size of 1 in the temporal dimension,
        # making it effectively a 2D conv applied to each frame.
        self.conv_spatial = nn.Conv3d(
            1, embed_dim,
            kernel_size=(1, self.patch_h, self.patch_w),
            stride=(1, self.patch_h, self.patch_w)
        )

    def forward(self, x):
        # x: [B, 1, T, H, W]
        x = self.conv_spatial(x)  # [B, E, T, H_p, W_p]
        
        # Reshape for per-frame token sequences
        # B, E, T, H_p, W_p -> B, T, E, H_p, W_p -> B, T, E, N_p -> B, T, N_p, E
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, E, H_p, W_p]
        x = x.flatten(3)              # [B, T, E, N_tokens_single_frame]
        x = x.transpose(2, 3)         # [B, T, N_tokens_single_frame, E]
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth ("DropPath").
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    """
    Simple MLP block with one hidden layer, GELU activation, dropout.
    """
    def __init__(self, embed_dim, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings.
    """
    def __init__(self, embed_dim, num_heads, attn_drop_rate=0., proj_drop_rate=0., causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # FIX: Add assertion to ensure head_dim is even for RoPE compatibility.
        assert self.head_dim % 2 == 0, "head_dim must be an even number for Rotary Position Embeddings"

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # RoPE
        cos, sin = self.rotary_emb(N, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with Rotary Position Embeddings.
    Queries from x, keys/values from context.
    """
    def __init__(self, embed_dim, num_heads, attn_drop_rate=0., proj_drop_rate=0., causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, context):
        B, N_q, C = x.shape
        _, N_kv, _ = context.shape

        q = self.q_proj(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        kv = self.kv_proj(context).reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        cos_q, sin_q = self.rotary_emb(N_q, x.device)
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        
        cos_k, sin_k = self.rotary_emb(N_kv, context.device)
        k = (k * cos_k) + (rotate_half(k) * sin_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed (queries can only attend to keys at earlier or same positions)
        if self.causal:
            # For cross-attention, we assume queries and keys have the same sequence length
            # and we want query at position i to only attend to keys at positions 0 to i
            mask = torch.tril(torch.ones(N_q, N_kv, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm -> MHSA -> DropPath -> Pre-LayerNorm -> MLP -> DropPath
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0., causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            causal=causal
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, drop_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VideoViT(nn.Module):
    """
    Vision Transformer for grayscale video clips with factorized patch embeddings.
    Includes additional temporal attention layers for global context.
    """
    def __init__(
        self,
        img_h=64, img_w=64, 
        frames_per_clip=16,
        patch_size_h=8, patch_size_w=8,
        embed_dim=768,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_drop_path_rate=0.1,
        encoder_num_temporal_layers=4
    ):
        super().__init__()
        self.patch_embed = FactorizedPatchEmbed(
            patch_size=(patch_size_h, patch_size_w),
            embed_dim=embed_dim
        )
        
        # Spatial transformer blocks
        dpr_spatial = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, encoder_num_heads,
                mlp_ratio, drop_rate,
                attn_drop_rate, dpr_spatial[i]
            ) for i in range(encoder_num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Temporal transformer blocks
        dpr_temporal = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_num_temporal_layers)]
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, encoder_num_heads,
                mlp_ratio, drop_rate,
                attn_drop_rate, dpr_temporal[i],
                causal=False
            ) for i in range(encoder_num_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        x = self.patch_embed(x)  # [B, T, N_tokens_single_frame, E]

        B, T, N, E = x.shape
        x = x.reshape(B * T, N, E)

        # Apply spatial transformer blocks to each frame's tokens independently
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x.reshape(B, T, N, E)
        
        # Pool the spatial patch tokens for each frame to get per-frame representations
        x = x.mean(dim=2)  # [B, T, E]
        
        # Apply temporal transformer blocks to learn global context across frames
        for blk in self.temporal_blocks:
            x = blk(x)
        x = self.temporal_norm(x)
        
        return x


class ConvEncoder(nn.Module):
    """
    Convolutional encoder
    
    Architecture:
    - Input: (B, 1, T, 64, 64) -> squeeze(1) -> (B, T, 64, 64)
    - 4 conv layers: 1→32→64→128→256 channels
    - Each conv: kernel_size=4, stride=2, padding=1, SiLU + LayerNorm
    - Final linear layer maps to latent_dim
    - Output: (B, T, latent_dim)
    """
    
    def __init__(self, latent_dim=64, input_channels=1, conv_channels=None, activation='silu', dropout_rate=0.0):
        super().__init__()
        
        # Use provided parameters or defaults
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        
        self.config = {
            'latent_dim': latent_dim,
            'input_channels': input_channels,
            'conv_channels': conv_channels,
            'activation': activation,
            'dropout_rate': dropout_rate
        }
        
        # Extract configuration
        self.latent_dim = self.config['latent_dim']
        input_channels = self.config['input_channels']
        conv_channels = self.config['conv_channels']
        
        # Get activation function
        if self.config['activation'].lower() == 'silu':
            activation_fn = nn.SiLU
        elif self.config['activation'].lower() == 'relu':
            activation_fn = nn.ReLU
        elif self.config['activation'].lower() == 'gelu':
            activation_fn = nn.GELU
        else:
            activation_fn = nn.SiLU  # Default fallback
        
        # Build CNN layers with LayerNorm after each conv
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_channels:
            # Conv layer
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            )
            self.conv_layers.append(conv)
            
            # LayerNorm for this layer's output channels
            # LayerNorm will be applied across the spatial dimensions
            self.layer_norms.append(nn.LayerNorm(out_channels))
            
            in_channels = out_channels
        
        # Activation function
        self.activation = activation_fn()
        
        # Dropout layer
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        
        # Calculate flattened size after convolutions
        # Input: 64x64, after 4 conv layers with stride=2: 64/2^4 = 4x4
        final_spatial_size = 64 // (2 ** len(conv_channels))  # 4x4
        flattened_size = conv_channels[-1] * final_spatial_size * final_spatial_size
        
        # Add LayerNorm before the final linear layer
        self.final_norm = nn.LayerNorm(flattened_size)
        
        # Final linear layer to latent dimension
        self.fc = nn.Linear(flattened_size, self.latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(m, nn.Conv2d):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (B, 1, T, 64, 64)
            
        Returns:
            Latent vectors of shape (B, T, latent_dim)
        """
        # Input shape: (B, 1, T, 64, 64)
        B, C, T, H, W = x.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert H == 64 and W == 64, f"Expected 64x64 images, got {H}x{W}"
        
        # Reshape to process all frames as a batch: (B, 1, T, 64, 64) -> (B*T, 1, 64, 64)
        x = x.reshape(B * T, C, H, W)
        
        # Pass through each conv layer with LayerNorm and Dropout
        for conv, layer_norm in zip(self.conv_layers, self.layer_norms):
            # Apply convolution
            x = conv(x)  # Shape: (B*T, C, H, W)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply LayerNorm: reshape from (B*T, C, H, W) to (B*T, H, W, C) for LayerNorm
            BT, C_out, H_out, W_out = x.shape
            x = x.permute(0, 2, 3, 1)  # (B*T, H, W, C)
            x = layer_norm(x)          # Apply LayerNorm on channel dimension
            x = x.permute(0, 3, 1, 2)  # Back to (B*T, C, H, W)
            
            # Apply dropout
            x = x.permute(0, 2, 3, 1)  # (B*T, H, W, C) for dropout
            x = self.dropout(x)
            x = x.permute(0, 3, 1, 2)  # Back to (B*T, C, H, W)
        
        # Flatten spatial dimensions
        x = x.flatten(start_dim=1)  # Shape: (B*T, C*H*W)
        
        # Apply final layer normalization
        x = self.final_norm(x)
        
        # Map to latent dimension
        latents = self.fc(x)  # Shape: (B*T, latent_dim)
        
        # Reshape back to (B, T, latent_dim)
        latents = latents.reshape(B, T, self.latent_dim)
        
        return latents

