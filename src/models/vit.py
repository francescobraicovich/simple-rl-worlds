from einops import repeat
import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from einops import rearrange
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Enable layer normalization for stability
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

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

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
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# Utility to pair ints
pair = lambda x: (x, x) if isinstance(x, int) else x

# Base Vision Transformer Class
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.01)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if num_classes > 0 else nn.Identity()

        self.apply(initialize_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Pool the output of the transformer
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else: # 'cls'
            x = x[:, 0] # Select the CLS token

        latent_representation = self.to_latent(x)
        
        # If mlp_head is Identity (num_classes=0), return latent representation
        # Otherwise, return classification logits
        return self.mlp_head(latent_representation)


# Refactored Video Vision Transformer using ViT for spatial processing
class ViTVideo(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_frames,
        num_classes,
        dim,
        # Spatial ViT parameters
        depth,
        heads,
        mlp_dim,
        # Temporal Transformer parameters
        temporal_depth = 2,
        temporal_heads = 8,
        temporal_mlp_dim = 256,
        # Common parameters
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        temporal_dropout=0.1
    ):
        super().__init__()

        # --- SPATIAL TRANSFORMER ---
        # Instantiate the ViT class to handle per-frame feature extraction.
        # We set num_classes=0 to ensure it returns the latent CLS token representation
        # for each frame, not classification logits.
        self.spatial_transformer = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=0, # Critical: ensures output is latent rep
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

        # --- TEMPORAL TRANSFORMER ---
        # video-CLS token + temporal positional embeddings
        self.video_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames + 1, dim) * 0.01)

        # Small transformer over [video_cls + per-frame CLS tokens]
        self.temporal_transformer = Transformer(
            dim=dim,
            depth=temporal_depth,
            heads=temporal_heads,
            dim_head=dim_head,
            mlp_dim=temporal_mlp_dim,
            dropout=temporal_dropout
        )

        # --- FINAL CLASSIFICATION HEAD ---
        self.to_latent = nn.Identity()
        self.mlp_head = (
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
            if num_classes > 0
            else nn.Identity()
        )
        
        self.apply(initialize_weights)

    def forward(self, video):
        """
        video: Tensor of shape [B, F, C, H, W]
        B: Batch size
        F: Number of frames
        C: Channels
        H: Height
        W: Width
        """
        B, F, C, H, W = video.shape

        # === 1) SPATIAL PROCESSING: Apply ViT to each frame ===
        # Reshape video into a batch of frames
        frames = video.view(B * F, C, H, W)

        # Get the CLS token representation for each frame
        # Output shape: [B*F, dim]
        frame_cls_tokens = self.spatial_transformer(frames)

        # Reshape back to separate batch and frame dimensions
        # Output shape: [B, F, dim]
        frame_cls_tokens = frame_cls_tokens.view(B, F, -1)

        # === 2) TEMPORAL PROCESSING: Apply Transformer over frame tokens ===
        # Prepend the video-level CLS token
        video_cls = repeat(self.video_cls, '() n d -> b n d', b=B)
        temporal_input = torch.cat((video_cls, frame_cls_tokens), dim=1) # Shape: [B, F+1, dim]

        # Add temporal positional embedding
        temporal_input += self.temporal_pos

        # Apply temporal attention
        temporal_output = self.temporal_transformer(temporal_input) # Shape: [B, F+1, dim]

        # === 3) FINAL PROJECTION ===
        # Extract the video-level CLS token for the final representation
        video_representation = temporal_output[:, 0]

        # Apply final classification head
        latent_rep = self.to_latent(video_representation)
        return self.mlp_head(latent_rep)