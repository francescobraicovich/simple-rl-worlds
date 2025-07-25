import torch
import torch.nn as nn
import torch.nn.functional as F
class MLPHistoryPredictor(nn.Module):
    """
    Multi-layer perceptron that predicts the next latent vector using the entire history
    of latent vectors and discrete actions.
    """
    
    def __init__(
        self,
        frames_per_clip: int = 4,
        latent_dim: int = 64,
        num_actions: int = 7,
        hidden_sizes: list = None,
        activation: str = 'silu',
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Set default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [512, 512]
        
        # Set parameters
        self.frames_per_clip = frames_per_clip
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
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
        num_actions: int = 7,
        hidden_sizes: list = None,
        activation: str = 'silu',
        dropout_rate: float = 0.1,
    ):
        """
        Create MLPHistoryPredictor by automatically inferring dimensions from encoder output.
        
        Args:
            encoder_output: Tensor of shape [B, T, E] from encoder
            num_actions: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
            dropout_rate: Dropout rate
            
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
