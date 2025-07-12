import torch
import torch.nn as nn

class RewardPredictor(nn.Module):
    def __init__(self, embedding_dim, internal_embedding_dim, num_heads=4):
        super(RewardPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.internal_embedding_dim = internal_embedding_dim

        self.proj_x1 = nn.Linear(embedding_dim, internal_embedding_dim)
        self.proj_x2 = nn.Linear(embedding_dim, internal_embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, internal_embedding_dim))
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=internal_embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(internal_embedding_dim, internal_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(internal_embedding_dim // 2, 1)
        )

    def forward(self, x1, x2):
        b, _, _ = x1.shape

        proj_x1 = self.proj_x1(x1)
        proj_x2 = self.proj_x2(x2)

        token_sequence = torch.cat([proj_x1, proj_x2], dim=1)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        
        attention_input = torch.cat([cls_tokens, token_sequence], dim=1)
        
        cls_output, _ = self.cross_attention(
            query=cls_tokens,
            key=attention_input,
            value=attention_input
        )
        
        reward = self.mlp_head(cls_output)
        
        return reward
