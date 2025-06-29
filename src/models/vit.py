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


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.01)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # The 'num_classes' argument suggests a classification head, which we might not need directly for representation.
        # We want the ViT to output the latent representation (either CLS token or mean pooled patch embeddings).
        # If 'num_classes' is for a final linear layer for classification, we can make it optional or remove it
        # if the primary use is feature extraction. For now, let's keep it as is, but our models
        # will likely use the output of self.to_latent.

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if num_classes > 0 else nn.Identity()  # Only add mlp_head if num_classes is positive

        self.apply(initialize_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Decide how to pool features
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:  # 'cls'
            x = x[:, 0]  # Take the CLS token

        latent_representation = self.to_latent(x)

        # If an mlp_head is defined (e.g. for supervised pre-training or if ViT is used for classification)
        # return self.mlp_head(latent_representation)
        # For our purposes, we usually need the latent_representation itself.
        # The calling model can decide if it wants to add more layers.
        # We will return the latent_representation directly.
        # If num_classes was specified > 0, the user might expect a classification output.
        # For JEPA and Encoder-Decoder, we need the latent vector.
        # Let's make it so that if num_classes=0, it returns the representation, else it passes through mlp_head.
        if self.mlp_head is nn.Identity():
            return latent_representation
        else:
            return self.mlp_head(latent_representation)

# Need to add 'repeat' for the cls_token. It's often part of einops.
# If it's not directly in the main einops, it might be in einops.layers or a separate import.
# Let's assume 'repeat' is available from einops for now.
# from einops import repeat # This should be at the top if not already there.
# It appears 'repeat' is not automatically imported with 'from einops import rearrange'.
# Let's ensure it's imported.


# Correcting imports for 'repeat'

# Final check on ViT class structure:
# The ViT class should take image_size, patch_size, dim (output latent dim), depth, heads, mlp_dim.
# The 'num_classes' can be set to the latent_dim if we want the mlp_head to project to that,
# or 0 if we want the raw pooled output.
# For our case, we often want the raw latent vector, so perhaps 'num_classes=0' is a good default for that.
# The current implementation has `mlp_head` which would be an identity if `num_classes=0`.
# This means it will return `latent_representation` as desired when `num_classes=0`.
