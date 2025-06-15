import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights


class MLPEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 image_size,  # e.g., (h, w) tuple or int if square
                 latent_dim,
                 num_hidden_layers=2,
                 hidden_dim=512,
                 activation_fn_str='relu',
                 dropout_rate=0.0):
        super().__init__()

        if isinstance(image_size, int):
            image_h, image_w = image_size, image_size
        else:
            image_h, image_w = image_size

        self.input_channels = input_channels
        self.image_h = image_h
        self.image_w = image_w
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if activation_fn_str == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn_str == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_fn_str}")

        input_dim = input_channels * image_h * image_w

        if input_dim == 0:
            raise ValueError(
                "Input dimension for MLP is 0. Check image_size and input_channels.")

        layers = []
        current_dim = input_dim

        if num_hidden_layers > 0:
            # First hidden layer
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation_fn)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

            # Subsequent hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation_fn)
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))

            # Output layer
            layers.append(nn.Linear(current_dim, latent_dim))
        else:  # No hidden layers, direct projection to latent_dim
            layers.append(nn.Linear(input_dim, latent_dim))

        self.mlp_net = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, img):
        # img: (batch, channels, height, width)
        batch_size = img.shape[0]

        # Validate input image dimensions against configured dimensions
        if img.shape[1] != self.input_channels or \
           img.shape[2] != self.image_h or \
           img.shape[3] != self.image_w:
            raise ValueError(
                f"Input image dimensions ({img.shape[1:]}) do not match "
                f"configured dimensions ({self.input_channels, self.image_h, self.image_w})."
            )

        x = img.view(batch_size, -1)  # Flatten

        expected_flattened_dim = self.input_channels * self.image_h * self.image_w
        if x.shape[1] != expected_flattened_dim:
            raise ValueError(
                f"Mismatch between expected flattened dimension ({expected_flattened_dim}) and "
                f"actual flattened dimension ({x.shape[1]}) in forward pass. "
                "This should not happen if input image dimensions are validated."
            )

        latent_representation = self.mlp_net(x)  # (batch, latent_dim)
        return latent_representation


if __name__ == '__main__':
    # Example Usage:
    bs = 4
    channels = 3
    img_size = 64
    ld = 128  # latent_dim
    dropout_val = 0.1

    try:
        mlp_encoder = MLPEncoder(input_channels=channels, image_size=img_size,
                                 latent_dim=ld, num_hidden_layers=2, hidden_dim=256, dropout_rate=dropout_val)
        print(f"MLP Encoder initialized: {mlp_encoder}")
        dummy_img = torch.randn(bs, channels, img_size, img_size)
        output = mlp_encoder(dummy_img)
        print(f"Output shape: {output.shape}")  # Expected: (bs, ld)
        assert output.shape == (bs, ld)

        mlp_encoder_no_hidden = MLPEncoder(
            input_channels=channels, image_size=img_size, latent_dim=ld, num_hidden_layers=0)
        print(
            f"MLP Encoder with no hidden layers initialized: {mlp_encoder_no_hidden}")
        output_no_hidden = mlp_encoder_no_hidden(dummy_img)
        print(f"Output shape with no hidden layers: {output_no_hidden.shape}")
        assert output_no_hidden.shape == (bs, ld)

        # Test with non-square image
        mlp_encoder_rect = MLPEncoder(
            input_channels=channels, image_size=(64, 32), latent_dim=ld)
        dummy_img_rect = torch.randn(bs, channels, 64, 32)
        output_rect = mlp_encoder_rect(dummy_img_rect)
        print(f"Output shape for rectangular image: {output_rect.shape}")
        assert output_rect.shape == (bs, ld)

    except ValueError as e:
        print(f"Error during MLPEncoder example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


class RewardPredictorMLP(nn.Module):
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

        if activation_fn_str == 'relu':
            activation_fn = nn.ReLU()
        elif activation_fn_str == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_fn_str}")

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation_fn)
            if self.dropout_rate > 0:
                self.layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

        # Output a single scalar reward
        self.layers.append(nn.Linear(current_dim, 1))
        self.apply(initialize_weights)

    def forward(self, x):
        # x: (batch_size, input_dim)
        for layer in self.layers:
            x = layer(x)
        return x  # (batch_size, 1)


if __name__ == '__main__':
    # Example Usage for MLPEncoder (existing):
    print("\n--- MLPEncoder Examples ---")
    bs = 4
    channels = 3
    img_size = 64
    ld = 128  # latent_dim
    dropout_val = 0.1 # Added for dropout example

    try:
        mlp_encoder = MLPEncoder(input_channels=channels, image_size=img_size,
                                 latent_dim=ld, num_hidden_layers=2, hidden_dim=256, dropout_rate=dropout_val)
        print(f"MLP Encoder initialized: {mlp_encoder}")
        dummy_img = torch.randn(bs, channels, img_size, img_size)
        output = mlp_encoder(dummy_img)
        print(f"Output shape: {output.shape}")  # Expected: (bs, ld)
        assert output.shape == (bs, ld)

        mlp_encoder_no_hidden = MLPEncoder(
            input_channels=channels, image_size=img_size, latent_dim=ld, num_hidden_layers=0)
        print(
            f"MLP Encoder with no hidden layers initialized: {mlp_encoder_no_hidden}")
        output_no_hidden = mlp_encoder_no_hidden(dummy_img)
        print(f"Output shape with no hidden layers: {output_no_hidden.shape}")
        assert output_no_hidden.shape == (bs, ld)

        # Test with non-square image
        mlp_encoder_rect = MLPEncoder(
            input_channels=channels, image_size=(64, 32), latent_dim=ld)
        dummy_img_rect = torch.randn(bs, channels, 64, 32)
        output_rect = mlp_encoder_rect(dummy_img_rect)
        print(f"Output shape for rectangular image: {output_rect.shape}")
        assert output_rect.shape == (bs, ld)

    except ValueError as e:
        print(f"Error during MLPEncoder example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during MLPEncoder example: {e}")

    # Example Usage for RewardPredictorMLP:
    print("\n--- RewardPredictorMLP Examples ---")
    batch_size = 8
    input_features = 256  # Example input feature dimension

    # Test case 1: With hidden layers
    dropout_val_reward = 0.05 # Example dropout for reward predictor
    try:
        reward_predictor = RewardPredictorMLP(input_dim=input_features, hidden_dims=[
                                              128, 64], activation_fn_str='relu', dropout_rate=dropout_val_reward)
        print(f"\nReward Predictor initialized: {reward_predictor}")
        dummy_input = torch.randn(batch_size, input_features)
        predicted_rewards = reward_predictor(dummy_input)
        # Expected: (batch_size, 1)
        print(f"Predicted rewards shape: {predicted_rewards.shape}")
        assert predicted_rewards.shape == (batch_size, 1)
        print(f"Predicted rewards sample: {predicted_rewards[0].item()}")
    except Exception as e:
        print(
            f"Error during RewardPredictorMLP example (with hidden layers): {e}")

    # Test case 2: No hidden layers (linear model)
    try:
        reward_predictor_linear = RewardPredictorMLP(
            input_dim=input_features, hidden_dims=[], activation_fn_str='relu')
        print(
            f"\nReward Predictor (linear) initialized: {reward_predictor_linear}")
        dummy_input_linear = torch.randn(batch_size, input_features)
        predicted_rewards_linear = reward_predictor_linear(dummy_input_linear)
        # Expected: (batch_size, 1)
        print(
            f"Predicted rewards shape (linear): {predicted_rewards_linear.shape}")
        assert predicted_rewards_linear.shape == (batch_size, 1)
        print(
            f"Predicted rewards sample (linear): {predicted_rewards_linear[0].item()}")
    except Exception as e:
        print(f"Error during RewardPredictorMLP example (linear): {e}")

    # Test case 3: With Batch Norm
    try:
        reward_predictor_bn = RewardPredictorMLP(input_dim=input_features, hidden_dims=[
                                                 128, 64], use_batch_norm=True, activation_fn_str='gelu', dropout_rate=dropout_val_reward)
        print(
            f"\nReward Predictor (with BatchNorm) initialized: {reward_predictor_bn}")
        dummy_input_bn = torch.randn(batch_size, input_features)
        # Set to eval mode if you want BatchNorm to use running stats, or train mode to update them
        # For a simple forward pass test, either is fine, but behavior differs during actual training.
        reward_predictor_bn.eval()  # Use running mean/var
        predicted_rewards_bn = reward_predictor_bn(dummy_input_bn)
        # Expected: (batch_size, 1)
        print(
            f"Predicted rewards shape (BatchNorm): {predicted_rewards_bn.shape}")
        assert predicted_rewards_bn.shape == (batch_size, 1)
        print(
            f"Predicted rewards sample (BatchNorm): {predicted_rewards_bn[0].item()}")

        # Test in training mode as well
        reward_predictor_bn.train()
        predicted_rewards_bn_train = reward_predictor_bn(
            dummy_input_bn)  # BN will use batch stats
        print(
            f"Predicted rewards shape (BatchNorm, train mode): {predicted_rewards_bn_train.shape}")
        assert predicted_rewards_bn_train.shape == (batch_size, 1)

    except Exception as e:
        print(f"Error during RewardPredictorMLP example (BatchNorm): {e}")
