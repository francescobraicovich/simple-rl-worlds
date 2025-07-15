import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MLPHistoryPredictor(nn.Module):
    """
    Multi-layer perceptron that predicts the next latent vector using the entire history
    of latent vectors and discrete actions.
    """
    
    # Default configuration
    default_config = {
        'frames_per_clip': 4,
        'latent_dim': 64,
        'num_actions': 7,
        'hidden_sizes': [512, 512],
        'activation': 'silu',
        'dropout_rate': 0.1
    }
    
    def __init__(
        self,
        frames_per_clip: int = None,
        latent_dim: int = None,
        num_actions: int = None,
        hidden_sizes: list = None,
        activation: str = 'silu',
        dropout_rate: float = 0.1,
        config_path: str = None,
    ):
        super().__init__()
        
        # Load configuration if provided
        if config_path is not None:
            from ..utils.config_utils import load_config
            config = load_config(config_path)
            config_num_actions = config.get('models', {}).get('predictor', {}).get('num_actions', 18)
        else:
            config_num_actions = 18
        
        # Set parameters with fallback to defaults
        self.frames_per_clip = frames_per_clip if frames_per_clip is not None else self.default_config['frames_per_clip']
        self.latent_dim = latent_dim if latent_dim is not None else self.default_config['latent_dim']
        self.num_actions = num_actions if num_actions is not None else config_num_actions
        
        if hidden_sizes is None:
            hidden_sizes = self.default_config['hidden_sizes']
        
        # Calculate input size: T * (E + A)
        input_size = self.frames_per_clip * (self.latent_dim + self.num_actions)
        
        # Choose activation function
        if activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, self.latent_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @classmethod
    def from_encoder_output(
        cls,
        encoder_output: torch.Tensor,
        num_actions: int = None,
        hidden_sizes: list = None,
        activation: str = 'silu',
        dropout_rate: float = 0.1,
        config_path: str = None
    ):
        """
        Create MLPHistoryPredictor by automatically inferring dimensions from encoder output.
        
        Args:
            encoder_output: Tensor of shape [B, T, E] from encoder
            num_actions: Number of discrete actions (if None, will try to load from config)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
            dropout_rate: Dropout rate
            config_path: Path to config file for num_actions
            
        Returns:
            MLPHistoryPredictor instance with inferred dimensions
        """
        if len(encoder_output.shape) != 3:
            raise ValueError(f"Expected encoder output shape [B, T, E], got {encoder_output.shape}")
        
        B, T, E = encoder_output.shape
        
        return cls(
            frames_per_clip=T,
            latent_dim=E,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_rate=dropout_rate,
            config_path=config_path
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, a: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: Per-frame latent vectors, shape [B, T, E]
            a: Discrete action indices, shape [B, T]
        Returns:
            x_pred: Predicted latent vector for the next frame, shape [B, 1, E]
        """
        B, T, E = x.shape

        # ensure a is long tensor
        if not a.is_floating_point():
            a = a.long()
        
        # Validate input dimensions
        if T != self.frames_per_clip:
            raise ValueError(f"Expected {self.frames_per_clip} frames, got {T}")
        if E != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {E}")
        
        # One-hot encode actions
        one_hot_a = F.one_hot(a, num_classes=self.num_actions).float()  # [B, T, A]
        
        # Concatenate latent vectors with one-hot actions
        x_cat = torch.cat([x, one_hot_a], dim=-1)  # [B, T, E + A]
        
        # Flatten sequence dimension
        x_flat = x_cat.view(B, T * (E + self.num_actions))  # [B, T * (E + A)]
        
        # Pass through MLP
        x_pred = self.mlp(x_flat)  # [B, E]
        
        # Reshape to [B, 1, E] to match expected output format
        x_pred = x_pred.unsqueeze(1)  # [B, 1, E]
        
        return x_pred
