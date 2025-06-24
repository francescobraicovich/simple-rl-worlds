import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights, print_num_parameters

class LookAheadRewardPredictorMLP(nn.Module):
    """
    Look-Ahead Reward Predictor (LARP) that uses future state predictions
    from world models to predict rewards.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list[int],
                 activation_fn_str: str = 'relu',
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        current_dim = input_dim

        # Define activation function
        if activation_fn_str.lower() == 'relu':
            activation_fn = nn.ReLU
        elif activation_fn_str.lower() == 'leaky_relu':
            activation_fn = nn.LeakyReLU
        elif activation_fn_str.lower() == 'tanh':
            activation_fn = nn.Tanh
        elif activation_fn_str.lower() == 'sigmoid':
            activation_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_str}")

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(activation_fn())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Output layer (predicts a single reward value)
        self.layers.append(nn.Linear(current_dim, 1))

        # Initialize weights
        self.apply(initialize_weights)
        print_num_parameters(self, check_total=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LARP MLP.
        Args:
            x: Input tensor containing concatenated features:
               - For Encoder-Decoder: [encoded_s_t, predicted_s_t_plus_1_flat, action_embedding]
               - For JEPA-Style Encoder-Decoder: [encoded_s_t, predicted_s_t_plus_1_flat, action_embedding, predictor_features]
               - For JEPA: [encoded_s_t, predicted_latent_s_t_plus_1, action_embedding]
        Returns:
            Predicted reward (single value).
        """
        for layer in self.layers:
            x = layer(x)
        return x
