import torch
import torch.nn as nn

# Reuse core building blocks (assumed available in the same package)
from .encoder import TransformerBlock

class LatentDynamicsPredictor(nn.Module):
    """
    Temporal Transformer to predict the next frame's latent representation.
    It takes a sequence of per-frame latent vectors [B, T, E], prepends an action token,
    and uses causal attention to predict the next frame.
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

        # Action embedding to be prepended as a separate token
        self.action_embed = nn.Embedding(num_actions, embed_dim)

        # Transformer blocks with causal attention
        dpr = [x.item() for x in torch.linspace(0, predictor_drop_path_rate, predictor_num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=predictor_num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                causal=True  # Enable causal masking
            )
            for i in range(predictor_num_layers)
        ])
        
        # Prediction head: LayerNorm + Linear
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, a: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: Per-frame latent vectors, shape [B, T, E] (from encoder)
            a: Discrete action indices, shape [B]
        Returns:
            x_pred: Predicted latent vector for the next frame, shape [B, 1, E]
        """
        B, T, E = x.shape
        
        # 1. Embed action and prepend as the first token
        a_emb = self.action_embed(a).unsqueeze(1)  # Shape: [B, 1, E]
        x = torch.cat((a_emb, x), dim=1)  # Shape: [B, T + 1, E]

        # 2. Process with Transformer blocks (using rotary embeddings and causal attention)
        for blk in self.blocks:
            x = blk(x)

        # 3. Extract the last token for prediction (represents next frame)
        last_token = x[:, -1:, :]  # Shape: [B, 1, E] - last token after causal attention
        
        # 4. Apply prediction head
        x_pred = self.prediction_head(last_token)  # Shape: [B, 1, E]
        
        return x_pred
