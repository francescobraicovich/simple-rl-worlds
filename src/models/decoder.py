# decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.fc_dec  = nn.Linear(latent_dim, 64 * 7 * 7)

        # *exact* mirror of conv3, conv2, conv1 — all padding=0
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(32,  4,  kernel_size=8, stride=4, padding=0)

    def forward(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 64, 7, 7)        # → (B,64,7,7)
        h = F.relu(self.deconv1(h))     # → (B,64,9,9)
        h = F.relu(self.deconv2(h))     # → (B,32,20,20)
        recon = self.deconv3(h)         # → (B, 4, 84,84)
        return recon


if __name__ == "__main__":
    # quick smoke test
    dec = Decoder(latent_dim=512)
    dummy_z = torch.randn(8, 512)
    out = dec(dummy_z)
    print("Decoder output shape:", out.shape)  # expect (8, 4, 84, 84)