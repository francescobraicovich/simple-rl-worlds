# encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    DQN‐style CNN encoder for Atari frames.
    Input: 4×84×84 tensor (stack of 4 grayscale frames)
    Output: latent vector of size `latent_dim`
    """
    def __init__(self, latent_dim: int = 512):
        super(Encoder, self).__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # Compute size after convs: 64 channels, 7×7 spatial
        self.fc_enc = nn.Linear(in_features=64 * 7 * 7, out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (B, 4, 84, 84)
        :return: Tensor of shape (B, latent_dim)
        """
        h = F.relu(self.conv1(x))   # → (B, 32, 20, 20)
        h = F.relu(self.conv2(h))   # → (B, 64, 9, 9)
        h = F.relu(self.conv3(h))   # → (B, 64, 7, 7)
        h = h.view(h.size(0), -1)   # → (B, 64*7*7)
        z = F.relu(self.fc_enc(h))  # → (B, latent_dim)
        return z


if __name__ == "__main__":
    # quick smoke test
    enc = Encoder(latent_dim=512)
    dummy = torch.randn(8, 4, 84, 84)
    out = enc(dummy)
    print("Encoder output shape:", out.shape)  # expect (8, 512)
