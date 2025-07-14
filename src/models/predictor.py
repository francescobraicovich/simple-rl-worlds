import torch
import torch.nn as nn

from .encoder import TransformerBlock, MultiHeadCrossAttention


class CrossAttentionLayer(nn.Module):
    """
    A layer that combines self-attention with cross-attention.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.self_attn = TransformerBlock(
            embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, drop_path_rate, causal=True
        )
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim, num_heads, attn_drop_rate, drop_rate, causal=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context):
        x = self.self_attn(x)
        # Apply cross-attention with a residual connection
        x = x + self.cross_attn(self.norm(x), context)
        return x


class LatentDynamicsPredictor(nn.Module):
    """
    Temporal Transformer to predict the next frame's latent representation.
    Uses cross-attention to condition on an action vector.
    """
    def __init__(
        self,
        frames_per_clip: int,
        embed_dim: int = 768,
        num_actions: int = 18,
        predictor_num_layers: int = 6,
        predictor_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        predictor_drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.frames_per_clip = frames_per_clip

        self.action_embed = nn.Embedding(num_actions, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, predictor_drop_path_rate, predictor_num_layers)]
        self.blocks = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=embed_dim,
                num_heads=predictor_num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i]
            )
            for i in range(predictor_num_layers)
        ])
        
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, a: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: Per-frame latent vectors, shape [B, T, E]
            a: Discrete action indices, shape [B, T]
        Returns:
            x_pred: Predicted latent vector for the next frame, shape [B, 1, E]
        """
        B, T, E = x.shape
        
        a_emb = self.action_embed(a)  # [B, T, E]

        # Process with cross-attention blocks
        for blk in self.blocks:
            x = blk(x, a_emb)

        # Extract the last token for prediction
        last_token = x[:, -1:, :]  # [B, 1, E]
        
        x_pred = self.prediction_head(last_token)
        
        return x_pred
