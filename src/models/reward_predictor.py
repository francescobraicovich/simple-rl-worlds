import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardPredictor(nn.Module):
    def __init__(self, embedding_dim, internal_embedding_dim, num_heads=4, 
                 num_attention_layers=1, mlp_hidden_layers=None, dropout=0.1, 
                 attention_dropout=0.1, use_layer_norm=True, activation="relu"):
        super(RewardPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.internal_embedding_dim = internal_embedding_dim
        self.num_attention_layers = num_attention_layers
        self.use_layer_norm = use_layer_norm

        # Input projections
        self.proj_x1 = nn.Linear(embedding_dim, internal_embedding_dim)
        self.proj_x2 = nn.Linear(embedding_dim, internal_embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, internal_embedding_dim))
        
        # Multiple cross-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=internal_embedding_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                batch_first=True
            ) for _ in range(num_attention_layers)
        ])

        # Layer normalization for attention layers
        if use_layer_norm:
            self.attention_layer_norms = nn.ModuleList([
                nn.LayerNorm(internal_embedding_dim) for _ in range(num_attention_layers)
            ])
        
        # Dropout for attention layers
        self.attention_dropout = nn.Dropout(dropout)

        # Build MLP head with configurable hidden layers
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [internal_embedding_dim // 2]
        
        mlp_layers = []
        prev_dim = internal_embedding_dim
        
        for hidden_dim in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(self._get_activation(activation))
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final output layer
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp_head = nn.Sequential(*mlp_layers)

    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()  # Default fallback

    def forward(self, x1, x2):
        b, _, _ = x1.shape

        # Project inputs to internal embedding dimension
        proj_x1 = self.proj_x1(x1)
        proj_x2 = self.proj_x2(x2)

        # Concatenate the two sequences
        token_sequence = torch.cat([proj_x1, proj_x2], dim=1)
        
        # Expand CLS tokens for the batch
        cls_tokens = self.cls_token.expand(b, -1, -1)
        
        # Prepare attention input (CLS token + token sequence)
        attention_input = torch.cat([cls_tokens, token_sequence], dim=1)
        
        # Process through multiple attention layers
        current_cls = cls_tokens
        current_context = attention_input
        
        for i in range(self.num_attention_layers):
            # Apply cross-attention
            attended_cls, _ = self.attention_layers[i](
                query=current_cls,
                key=current_context,
                value=current_context
            )
            
            # Residual connection and layer norm
            if self.use_layer_norm:
                current_cls = self.attention_layer_norms[i](current_cls + self.attention_dropout(attended_cls))
            else:
                current_cls = current_cls + self.attention_dropout(attended_cls)
            
            # Update context for next layer (include updated CLS token)
            current_context = torch.cat([current_cls, token_sequence], dim=1)
        
        # Pass through MLP head to get final reward prediction
        reward = self.mlp_head(current_cls)
        
        return reward
