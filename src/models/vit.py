# Video Vision Transformer tailored for Reinforcement‑Learning with frame‑stacked inputs (grayscale)
# -----------------------------------------------------------------------------------
# - Accepts a video tensor of shape [B, T, H, W] (T ≤ num_frames), no explicit channel dim.
# - Uses a lightweight Conv2d patch embed like modern ViTs.
# - Performs spatial self‑attention *per frame* (weights shared across time).
# - Applies a compact temporal transformer on frame‑level tokens.
# - Classification head with `num_classes` to output logits.

import torch
import torch.nn as nn
from einops import repeat, rearrange
from src.utils.weight_init import initialize_weights

# helpers

class ViT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise DeprecationWarning("ViT is deprecated, use ViTVideo instead.")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity())

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTVideo(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        temporal_depth=2,
        temporal_heads=8,
        temporal_mlp_dim=256,
        pool='cls',
        channels=1,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        temporal_dropout=0.1
    ):
        super().__init__()
        # Spatial embedding via Conv2d patch projection (grayscale)
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, 'Image must be divisible by patch size'
        num_patches = (ih // ph) * (iw // pw)

        self.patch_embed = nn.Conv2d(
            1, dim, kernel_size=(ph, pw), stride=(ph, pw)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.01)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.dropout = nn.Dropout(emb_dropout)

        # Shared spatial transformer
        self.spatial_transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout
        )
        self.pool = pool

        # Temporal transformer setup
        self.video_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames + 1, dim) * 0.01)
        self.temporal_transformer = Transformer(
            dim=dim, depth=temporal_depth, heads=temporal_heads,
            dim_head=dim_head, mlp_dim=temporal_mlp_dim, dropout=temporal_dropout
        )

        # Classification head
        self.to_latent = nn.Identity()

        self.apply(initialize_weights)

    def forward(self, video):
        # video: [B, T, H, W] (no channel dim)
        B, T, H, W = video.shape
        # add channel dimension for patch embedding
        x = video.view(B * T, 1, H, W)
        x = self.patch_embed(x)      # [B*T, dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, dim]

        # Add cls token & pos
        cls = repeat(self.cls_token, '() n d -> b n d', b=B * T)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)

        # Spatial transformer
        x = self.spatial_transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]
        # reshape to [B, T, dim]
        x = x.view(B, T, -1)

        # Temporal sequence
        video_cls = repeat(self.video_cls, '() n d -> b n d', b=B)
        y = torch.cat([video_cls, x], dim=1)
        y = y + self.temporal_pos[:, :T+1]
        y = self.temporal_transformer(y)

        # take video cls
        v = y[:, 0]
        return self.to_latent(v)
