import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from src.utils.weight_init import initialize_weights

class StateDecoder(nn.Module):
    def __init__(self,
                 input_latent_dim: int,
                 decoder_dim: int,
                 decoder_depth: int,
                 decoder_heads: int,
                 decoder_mlp_dim: int,
                 output_channels: int,
                 output_image_size: tuple[int, int], # (height, width)
                 decoder_dropout: float = 0.0,
                 decoder_patch_size: int = 8):
        super().__init__()

        self.output_channels = output_channels
        self.output_image_h, self.output_image_w = output_image_size
        self.decoder_patch_size = decoder_patch_size

        if self.output_image_h % decoder_patch_size != 0 or self.output_image_w % decoder_patch_size != 0:
            raise ValueError("Output image dimensions must be divisible by the decoder patch size.")

        self.output_num_patches_h = self.output_image_h // decoder_patch_size
        self.output_num_patches_w = self.output_image_w // decoder_patch_size
        self.num_output_patches = self.output_num_patches_h * self.output_num_patches_w

        output_patch_dim = output_channels * decoder_patch_size * decoder_patch_size

        # Project the JEPA predictor's output to decoder dimension
        self.decoder_input_projection = nn.Linear(input_latent_dim, decoder_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_mlp_dim,
            dropout=decoder_dropout,
            batch_first=True # Important: batch dimension first
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=decoder_depth
        )

        # Learnable query tokens for the decoder (one set of queries for all patches)
        self.decoder_query_tokens = nn.Parameter(torch.randn(1, self.num_output_patches, decoder_dim) * 0.02) # Standardized init

        # Map decoder output to pixel values for each patch
        self.to_pixels = nn.Linear(decoder_dim, output_patch_dim)

        # Rearrange patches back into an image
        self.patch_to_image = Rearrange(
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            p1=decoder_patch_size, p2=decoder_patch_size,
            h=self.output_num_patches_h, w=self.output_num_patches_w,
            c=output_channels
        )
        self.apply(initialize_weights)

    def forward(self, jepa_predictor_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the JEPA State Decoder.
        Args:
            jepa_predictor_embedding: Tensor of shape (b, input_latent_dim) from JEPA's predictor.
        Returns:
            predicted_next_state_image: Tensor of shape (b, output_channels, output_image_h, output_image_w).
        """
        batch_size = jepa_predictor_embedding.shape[0]

        # 1. Project predictor embedding to decoder dimension to serve as memory
        # Shape: (b, input_latent_dim) -> (b, decoder_dim)
        decoder_memory = self.decoder_input_projection(jepa_predictor_embedding)
        # Unsqueeze to make it (b, 1, decoder_dim) to act as a single memory item for the transformer
        decoder_memory = decoder_memory.unsqueeze(1)

        # 2. Prepare query tokens for the decoder
        # Repeat query tokens for the batch size: (1, num_output_patches, decoder_dim) -> (b, num_output_patches, decoder_dim)
        query_tokens = self.decoder_query_tokens.repeat(batch_size, 1, 1)

        # 3. Pass through Transformer Decoder
        # query_tokens (tgt): (b, num_output_patches, decoder_dim)
        # decoder_memory (memory): (b, 1, decoder_dim)
        # Output shape: (b, num_output_patches, decoder_dim)
        decoded_representation = self.transformer_decoder(tgt=query_tokens, memory=decoder_memory)

        # 4. Map to pixel values
        # Shape: (b, num_output_patches, decoder_dim) -> (b, num_output_patches, output_patch_dim)
        pixel_patches = self.to_pixels(decoded_representation)

        # 5. Reshape patches to image
        # Shape: (b, num_output_patches, output_patch_dim) -> (b, output_channels, output_image_h, output_image_w)
        predicted_next_state_image = self.patch_to_image(pixel_patches)

        return predicted_next_state_image

if __name__ == '__main__':
    # Example Usage (for testing the decoder directly)
    batch_size_test = 4
    latent_dim_test = 256 # Example latent dim from JEPA predictor
    decoder_dim_test = 128
    decoder_depth_test = 3
    decoder_heads_test = 4
    decoder_mlp_dim_test = 512
    output_channels_test = 3
    image_size_test = (64, 64)
    patch_size_test = 8

    # Instantiate the decoder
    jepa_decoder = StateDecoder(
        input_latent_dim=latent_dim_test,
        decoder_dim=decoder_dim_test,
        decoder_depth=decoder_depth_test,
        decoder_heads=decoder_heads_test,
        decoder_mlp_dim=decoder_mlp_dim_test,
        output_channels=output_channels_test,
        output_image_size=image_size_test,
        decoder_patch_size=patch_size_test
    )

    # Create a dummy input tensor (simulating output from JEPA predictor)
    dummy_predictor_output = torch.randn(batch_size_test, latent_dim_test)

    # Get the predicted next state image
    predicted_image = jepa_decoder(dummy_predictor_output)

    print("JEPA State Decoder initialized.")
    print(f"Input predictor embedding shape: {dummy_predictor_output.shape}")
    print(f"Output predicted image shape: {predicted_image.shape}")

    # Expected output shape: (batch_size_test, output_channels_test, image_size_test[0], image_size_test[1])
    assert predicted_image.shape == (batch_size_test, output_channels_test, image_size_test[0], image_size_test[1])
    print("Test passed: Output shape is correct.")

    # Test with non-square image
    image_size_test_rect = (48, 64)
    patch_size_test_rect = 8
    jepa_decoder_rect = StateDecoder(
        input_latent_dim=latent_dim_test,
        decoder_dim=decoder_dim_test,
        decoder_depth=decoder_depth_test,
        decoder_heads=decoder_heads_test,
        decoder_mlp_dim=decoder_mlp_dim_test,
        output_channels=output_channels_test,
        output_image_size=image_size_test_rect,
        decoder_patch_size=patch_size_test_rect
    )
    dummy_predictor_output_rect = torch.randn(batch_size_test, latent_dim_test)
    predicted_image_rect = jepa_decoder_rect(dummy_predictor_output_rect)
    print(f"Output predicted image shape (rectangular): {predicted_image_rect.shape}")
    assert predicted_image_rect.shape == (batch_size_test, output_channels_test, image_size_test_rect[0], image_size_test_rect[1])
    print("Test passed: Rectangular output shape is correct.")

    # Test invalid patch size
    try:
        StateDecoder(
            input_latent_dim=latent_dim_test,
            decoder_dim=decoder_dim_test,
            decoder_depth=decoder_depth_test,
            decoder_heads=decoder_heads_test,
            decoder_mlp_dim=decoder_mlp_dim_test,
            output_channels=output_channels_test,
            output_image_size=(64, 60), # 60 is not divisible by 8
            decoder_patch_size=patch_size_test
        )
    except ValueError as e:
        print(f"Caught expected error for invalid patch size: {e}")

    print("JEPAStateDecoder basic tests completed.")
