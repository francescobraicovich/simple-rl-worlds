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
    (Corrected and Modified Version)
    """
    def __init__(
        self,
        # img_h and img_w are not needed for the conv patcher but kept for clarity
        img_h=64, img_w=64, 
        frames_per_clip=16, # Not used in this implementation but good for context
        patch_size_h=8, patch_size_w=8, # Removed patch_size_t
        embed_dim=768,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_drop_path_rate=0.1
    ):
        super().__init__()
        # Corrected call to FactorizedPatchEmbed
        self.patch_embed = FactorizedPatchEmbed(
            patch_size=(patch_size_h, patch_size_w),
            embed_dim=embed_dim
        )
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, encoder_num_heads,
                mlp_ratio, drop_rate,
                attn_drop_rate, dpr[i]
            ) for i in range(encoder_num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        x = self.patch_embed(x)  # [B, T, N_tokens_single_frame, E]

        # Reshape to process all frames as a single batch
        B, T, N, E = x.shape
        x = x.reshape(B * T, N, E)

        # Apply transformer blocks to each frame's tokens independently
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Reshape back to per-frame token sequences
        x = x.reshape(B, T, N, E)
        
        # MODIFICATION: Pool the spatial patch tokens for each frame
        # (B, T, N, E) -> (B, T, E)
        x = x.mean(dim=2)
        
        return x