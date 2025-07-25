import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPretrainPredictor(nn.Module):
    """
    MLP predictor for JEPA pretraining that predicts masked parts of encoded frames.
    
    Takes encoded frames of shape [B, latent_dim] and predicts the masked portions
    for self-supervised representation learning.
    
    Input: [B, latent_dim] - encoded frame from JEPA encoder
    Output: [B, latent_dim] - prediction of masked parts
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_sizes: list = None,
        activation: str = 'silu',
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the JEPA pretrain predictor.
        
        Args:
            latent_dim: Dimension of the input and output latent vectors
            hidden_sizes: List of hidden layer sizes. If None, uses default [512, 512]
            activation: Activation function ('silu', 'relu', 'gelu')
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Set default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [512, 512]
        
        # Store parameters
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
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
        prev_size = latent_dim
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(prev_size, latent_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using Xavier uniform for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP predictor.
        
        Args:
            x: Input tensor of shape [B, latent_dim] - encoded frame
            
        Returns:
            prediction: Output tensor of shape [B, latent_dim] - predicted masked parts
        """
        # Validate input shape
        if len(x.shape) != 2:
            raise ValueError(f"Expected input shape [B, latent_dim], got {x.shape}")
        
        batch_size, input_dim = x.shape
        if input_dim != self.latent_dim:
            raise ValueError(f"Expected input dimension {self.latent_dim}, got {input_dim}")
        
        # Pass through MLP
        prediction = self.mlp(x)
        
        return prediction
    
    @classmethod
    def from_config(cls, config: dict):
        """
        Create JEPAPretrainPredictor from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing model parameters
            
        Returns:
            JEPAPretrainPredictor instance
        """
        latent_dim = config.get('latent_dim', 512)
        predictor_config = config.get('models', {}).get('jepa_pretrain_predictor', {})
        
        return cls(
            latent_dim=latent_dim,
            hidden_sizes=predictor_config.get('hidden_sizes', [512, 512]),
            activation=predictor_config.get('activation', 'silu'),
            dropout_rate=predictor_config.get('dropout_rate', 0.1),
        )
