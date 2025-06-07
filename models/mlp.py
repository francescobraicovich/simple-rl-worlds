import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 image_size, # e.g., (h, w) tuple or int if square
                 latent_dim,
                 num_hidden_layers=2,
                 hidden_dim=512,
                 activation_fn_str='relu'):
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

        if activation_fn_str == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn_str == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_str}")

        input_dim = input_channels * image_h * image_w

        if input_dim == 0:
            raise ValueError("Input dimension for MLP is 0. Check image_size and input_channels.")

        layers = []
        current_dim = input_dim

        if num_hidden_layers > 0:
            # First hidden layer
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation_fn)
            current_dim = hidden_dim

            # Subsequent hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation_fn)

            # Output layer
            layers.append(nn.Linear(current_dim, latent_dim))
        else: # No hidden layers, direct projection to latent_dim
            layers.append(nn.Linear(input_dim, latent_dim))

        self.mlp_net = nn.Sequential(*layers)

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

        x = img.view(batch_size, -1) # Flatten

        expected_flattened_dim = self.input_channels * self.image_h * self.image_w
        if x.shape[1] != expected_flattened_dim:
             raise ValueError(
                f"Mismatch between expected flattened dimension ({expected_flattened_dim}) and "
                f"actual flattened dimension ({x.shape[1]}) in forward pass. "
                "This should not happen if input image dimensions are validated."
            )

        latent_representation = self.mlp_net(x) # (batch, latent_dim)
        return latent_representation

if __name__ == '__main__':
    # Example Usage:
    bs = 4
    channels = 3
    img_size = 64
    ld = 128 # latent_dim

    try:
        mlp_encoder = MLPEncoder(input_channels=channels, image_size=img_size, latent_dim=ld, num_hidden_layers=2, hidden_dim=256)
        print(f"MLP Encoder initialized: {mlp_encoder}")
        dummy_img = torch.randn(bs, channels, img_size, img_size)
        output = mlp_encoder(dummy_img)
        print(f"Output shape: {output.shape}") # Expected: (bs, ld)
        assert output.shape == (bs, ld)

        mlp_encoder_no_hidden = MLPEncoder(input_channels=channels, image_size=img_size, latent_dim=ld, num_hidden_layers=0)
        print(f"MLP Encoder with no hidden layers initialized: {mlp_encoder_no_hidden}")
        output_no_hidden = mlp_encoder_no_hidden(dummy_img)
        print(f"Output shape with no hidden layers: {output_no_hidden.shape}")
        assert output_no_hidden.shape == (bs, ld)

        # Test with non-square image
        mlp_encoder_rect = MLPEncoder(input_channels=channels, image_size=(64,32), latent_dim=ld)
        dummy_img_rect = torch.randn(bs, channels, 64, 32)
        output_rect = mlp_encoder_rect(dummy_img_rect)
        print(f"Output shape for rectangular image: {output_rect.shape}")
        assert output_rect.shape == (bs, ld)

    except ValueError as e:
        print(f"Error during MLPEncoder example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
