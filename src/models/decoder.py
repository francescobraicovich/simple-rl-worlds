import math
import torch
import torch.nn as nn
from typing import Optional

class HybridConvTransformerDecoder(nn.Module):
    """
    Simplified Convolutional Decoder for grayscale frame reconstruction.
    Accepts a single predicted latent token and reconstructs a complete frame
    using transposed convolutions.
    """
    def __init__(
        self,
        img_h: int = 64,
        img_w: int = 64,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        patch_size_h: int = 8,
        patch_size_w: int = 8,
        **kwargs # Ignore other parameters
    ):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # --- Calculate Upsampling ---
        # We start from a 1x1 spatial dimension and need to reach H_p x W_p
        self.H_p = img_h // patch_size_h
        self.W_p = img_w // patch_size_w

        # The number of 2x upsampling steps
        num_upsampling_blocks = int(math.log2(self.H_p))
        if not math.log2(self.H_p).is_integer() or not math.log2(self.W_p).is_integer():
            raise ValueError("Patch sizes must result in a power-of-2 upsampling factor.")

        # --- Layers ---
        # Project the latent token to a small spatial grid with decoder_embed_dim channels
        self.proj = nn.Linear(embed_dim, decoder_embed_dim * 1 * 1)

        # Build the upsampling blocks
        blocks = []
        in_channels = decoder_embed_dim
        for i in range(num_upsampling_blocks):
            out_channels = decoder_embed_dim // 2 if i < num_upsampling_blocks - 1 else decoder_embed_dim
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels)
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        # Final convolution to get to the right patch size and then to a single channel image
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)


    def forward(self, latent_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_token: [B, 1, embed_dim] - single predicted latent token
        Returns:
            recon: [B, 1, 1, img_h, img_w] - reconstructed frame
        """
        B, _, E = latent_token.shape
        x = self.proj(latent_token.squeeze(1))
        x = x.view(B, self.decoder_embed_dim, 1, 1) # Reshape to a 1x1 spatial grid

        x = self.blocks(x) # Upsample to H_p x W_p

        # At this point, x has shape [B, decoder_embed_dim, H_p, W_p]
        # We need to get to [B, 1, img_h, img_w]
        # We can do this with a final transposed convolution
        
        recon = nn.functional.interpolate(x, size=(self.img_h, self.img_w), mode='bilinear', align_corners=False)
        recon = self.final_conv(recon)
        
        return recon.unsqueeze(1) # Add temporal dimension
