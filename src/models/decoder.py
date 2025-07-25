# decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Symmetric transpose‐conv decoder for reconstructing Atari frames.
    Input: latent vector of size `latent_dim`
    Output: 4×84×84 tensor (reconstructed stack of frames)
    """
    def __init__(self, latent_dim: int = 512):
        super(Decoder, self).__init__()
        # Project up to conv feature map size
        self.fc_dec  = nn.Linear(in_features=latent_dim, out_features=64 * 7 * 7)
        # Transposed convolutions (mirror of conv1–3)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=4,
                                          kernel_size=8, stride=4, padding=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: Tensor of shape (B, latent_dim)
        :return: Tensor of shape (B, 4, 84, 84)
        """
        h = F.relu(self.fc_dec(z))      # → (B, 64*7*7)
        h = h.view(-1, 64, 7, 7)        # → (B, 64, 7, 7)
        h = F.relu(self.deconv1(h))     # → (B, 64, 9, 9)
        h = F.relu(self.deconv2(h))     # → (B, 32, 20, 20)
        recon = self.deconv3(h)         # → (B, 4, 84, 84)
        return recon


if __name__ == "__main__":
    # quick smoke test
    dec = Decoder(latent_dim=512)
    dummy_z = torch.randn(8, 512)
    out = dec(dummy_z)
    print("Decoder output shape:", out.shape)  # expect (8, 4, 84, 84)