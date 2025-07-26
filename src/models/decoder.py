import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 512, n_segment: int = 4):
        super().__init__()
        self.n_segment = n_segment
        self.fc_dec  = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, n_segment, kernel_size=8, stride=4, padding=0)

    def inverse_shift(self, x):
        B, C, H, W = x.size()
        n = C // 4
        if n == 0:
            return x
        x1 = x[:, :n]        # originally backward-shifted group
        x2 = x[:, n:2*n]     # originally forward-shifted group
        x3 = x[:, 2*n:]      # static group
        # undo backward shift: shift forward
        x1 = F.pad(x1, (0,0,0,0,0,1))[:, 1:]
        # undo forward shift: shift backward
        x2 = F.pad(x2, (0,0,0,0,1,0))[:, :n]
        return torch.cat([x1, x2, x3], dim=1)

    def forward(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.deconv1(h))     # -> (B,64,9,9)
        h = self.inverse_shift(h)
        h = F.relu(self.deconv2(h))     # -> (B,32,20,20)
        h = self.inverse_shift(h)
        recon = self.deconv3(h)         # -> (B,4,84,84)
        return recon

if __name__ == "__main__":
    dec = Decoder(latent_dim=512, n_segment=4)
    dummy_z = torch.randn(8, 512)
    out = dec(dummy_z)
    print("Decoder output shape:", out.shape)  # expect (8, 4, 84, 84)
