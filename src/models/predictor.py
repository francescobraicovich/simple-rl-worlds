import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse core building blocks (assumed available in the same package)
from .encoder import DropPath, MultiHeadSelfAttention, MLP, TransformerBlock

class LatentDynamicsPredictor(nn.Module):
    """
    Transformer-based predictor for next-frame latent dynamics conditioned on actions.

    Args:
        embed_dim (int): Dimensionality of token embeddings.
        num_actions (int): Number of discrete actions for conditioning.
        predictor_num_layers (int): Number of Transformer blocks.
        predictor_num_heads (int): Number of attention heads per block.
        mlp_ratio (float): Ratio for hidden dimension in MLP (mlp_ratio * embed_dim).
        drop_rate (float): Dropout rate for MLP and projection layers.
        attn_drop_rate (float): Dropout rate for attention weights.
        predictor_drop_path_rate (float): Maximum stochastic depth rate.
    """
    def __init__(
        self,
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
        # Action embedding table
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        # Build transformer blocks with stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, predictor_drop_path_rate, predictor_num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=predictor_num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i]
            )
            for i in range(predictor_num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, a: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: Latent tokens for current frame, shape [B, N_tokens, embed_dim]
            a: Discrete action indices, shape [B]
        Returns:
            x_pred: Predicted next-frame latent tokens, shape [B, N_tokens, embed_dim]
        """
        # Embed actions and broadcast
        B, N, E = x.shape
        a_emb = self.action_embed(a)                  # [B, embed_dim]
        a_emb = a_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, embed_dim]
        x = x + a_emb                                 # Condition tokens on action

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # Final normalization
        x_pred = self.norm(x)
        return x_pred
