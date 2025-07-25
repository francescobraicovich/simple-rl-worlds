import torch
import torch.nn as nn
import torch.nn.functional as F
class MLPHistoryPredictor(nn.Module):
    """
    Multi-layer perceptron that predicts the next latent vector using the entire history
    of latent vectors (concatenated into one vector) and the last discrete action.
    """
    
    def __init__(
        self,
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
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # Calculate input size: E (latent vector) + A (one-hot action)
        input_size = self.latent_dim + self.num_actions
        
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
            encoder_output: Tensor of shape [B, E] from encoder (all frames concatenated)
            num_actions: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
            dropout_rate: Dropout rate
            
        Returns:
            MLPHistoryPredictor instance with inferred dimensions
        """
        if len(encoder_output.shape) != 2:
            raise ValueError(f"Expected encoder output shape [B, E], got {encoder_output.shape}")
        
        B, E = encoder_output.shape
        
        return cls(
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
            x: Concatenated latent vectors, shape [B, E]
            a: Discrete action index, shape [B]
        Returns:
            x_pred: Predicted latent vector for the next frame, shape [B, E]
        """
        B, E = x.shape

        # ensure a is long tensor
        if not a.is_floating_point():
            a = a.long()
        
        # Validate input dimensions
        if E != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {E}")
        
        # One-hot encode actions
        one_hot_a = F.one_hot(a, num_classes=self.num_actions).float()  # [B, A]
        
        # Concatenate latent vector with one-hot action
        x_cat = torch.cat([x, one_hot_a], dim=-1)  # [B, E + A]
        
        # Pass through MLP
        x_pred = self.mlp(x_cat)  # [B, E]
        
        return x_pred
