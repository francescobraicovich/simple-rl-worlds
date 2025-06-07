import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 image_size,  # e.g., (h, w) tuple or int if square
                 latent_dim,
                 num_conv_layers=3,
                 base_filters=32,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 activation_fn_str='relu',
                 fc_hidden_dim=None):
        super().__init__()

        if isinstance(image_size, int):
            image_h, image_w = image_size, image_size
        else:
            image_h, image_w = image_size

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_conv_layers = num_conv_layers
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if activation_fn_str == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn_str == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_fn_str}")

        conv_layers = []
        current_channels = input_channels
        current_h, current_w = image_h, image_w

        for i in range(num_conv_layers):
            out_channels = base_filters * (2**i)
            conv_layers.append(
                nn.Conv2d(current_channels, out_channels,
                          kernel_size, stride, padding)
            )
            conv_layers.append(self.activation_fn)
            # conv_layers.append(nn.BatchNorm2d(out_channels)) # Optional: BatchNorm
            # conv_layers.append(nn.MaxPool2d(2, 2)) # Optional: MaxPool, adjust stride if used

            current_channels = out_channels
            # Calculate output size after conv
            current_h = (current_h + 2 * padding - kernel_size) // stride + 1
            current_w = (current_w + 2 * padding - kernel_size) // stride + 1

            if current_h <= 0 or current_w <= 0:
                raise ValueError(
                    f"Image dimensions became non-positive after {i+1} conv layers. "
                    f"Current H: {current_h}, W: {current_w}. "
                    "Try fewer layers, smaller kernel, larger stride, or different padding."
                )

        self.conv_net = nn.Sequential(*conv_layers)

        # Calculate the flattened size after conv layers
        self.flattened_size = current_channels * current_h * current_w

        if self.flattened_size == 0:
            raise ValueError(
                "Flattened size is 0 after conv layers. Check network parameters.")

        if fc_hidden_dim:
            self.fc_net = nn.Sequential(
                nn.Linear(self.flattened_size, fc_hidden_dim),
                self.activation_fn,
                nn.Linear(fc_hidden_dim, latent_dim)
            )
        else:
            self.fc_net = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, img):
        # img: (batch, channels, height, width)
        x = self.conv_net(img)
        x = x.view(x.size(0), -1)  # Flatten

        if x.shape[1] != self.flattened_size:
            # This can happen if image_size used for calculation differs from actual input image size.
            # It's better to calculate flattened_size dynamically in forward if input size can vary.
            # For now, assume fixed input size as per constructor.
            raise ValueError(
                f"Mismatch between calculated flattened size ({self.flattened_size}) and "
                f"actual flattened size ({x.shape[1]}) in forward pass. "
                "Ensure image_size parameter matches input image dimensions."
            )

        latent_representation = self.fc_net(x)  # (batch, latent_dim)
        return latent_representation


if __name__ == '__main__':
    # Example Usage:
    bs = 4
    channels = 3
    img_size = 64
    ld = 128  # latent_dim

    try:
        cnn_encoder = CNNEncoder(input_channels=channels, image_size=img_size,
                                 latent_dim=ld, num_conv_layers=3, base_filters=32)
        print(f"CNN Encoder initialized: {cnn_encoder}")
        dummy_img = torch.randn(bs, channels, img_size, img_size)
        output = cnn_encoder(dummy_img)
        print(f"Output shape: {output.shape}")  # Expected: (bs, ld)
        assert output.shape == (bs, ld)

        cnn_encoder_fc_hidden = CNNEncoder(
            input_channels=channels, image_size=img_size, latent_dim=ld, fc_hidden_dim=256)
        print(
            f"CNN Encoder with fc_hidden_dim initialized: {cnn_encoder_fc_hidden}")
        output_fc_hidden = cnn_encoder_fc_hidden(dummy_img)
        print(f"Output shape with fc_hidden_dim: {output_fc_hidden.shape}")
        assert output_fc_hidden.shape == (bs, ld)

        # Test with non-square image
        cnn_encoder_rect = CNNEncoder(
            input_channels=channels, image_size=(64, 96), latent_dim=ld)
        dummy_img_rect = torch.randn(bs, channels, 64, 96)
        output_rect = cnn_encoder_rect(dummy_img_rect)
        print(f"Output shape for rectangular image: {output_rect.shape}")
        assert output_rect.shape == (bs, ld)

        # Test case that might lead to small feature map
        # cnn_encoder_deep = CNNEncoder(input_channels=channels, image_size=32, latent_dim=ld, num_conv_layers=4)
        # dummy_img_32 = torch.randn(bs, channels, 32, 32)
        # output_deep = cnn_encoder_deep(dummy_img_32)
        # print(f"Output shape for deep encoder on small image: {output_deep.shape}")
        # assert output_deep.shape == (bs, ld)

    except ValueError as e:
        print(f"Error during CNNEncoder example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
