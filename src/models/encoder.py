import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 512, n_segment: int = 4):
        super().__init__()
        self.n_segment = n_segment
        # input channels = n_segment (temporal frames)
        self.conv1 = nn.Conv2d(n_segment, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc_enc = nn.Linear(64 * 7 * 7, latent_dim)

    def temporal_shift(self, x):
        B, C, H, W = x.size()
        # quarter of channels for shift groups
        n = C // 4
        if n == 0:
            return x
        x1 = x[:, :n]        # to shift backward (t -> t+1)
        x2 = x[:, n:2*n]     # to shift forward  (t -> t-1)
        x3 = x[:, 2*n:]      # static
        # backward shift: pad front then crop
        x1 = F.pad(x1, (0,0,0,0,1,0))[:, :n]
        # forward shift: pad back then crop
        x2 = F.pad(x2, (0,0,0,0,0,1))[:, 1:]
        return torch.cat([x1, x2, x3], dim=1)

    def forward(self, x):
        # x shape: (B, 4, 84, 84)
        x = self.temporal_shift(x)
        x = F.relu(self.conv1(x))  # -> (B,32,20,20)
        x = self.temporal_shift(x)
        x = F.relu(self.conv2(x))  # -> (B,64,9,9)
        x = F.relu(self.conv3(x))  # -> (B,64,7,7)
        z = self.fc_enc(x.view(x.size(0), -1))
        return z

if __name__ == "__main__":
    enc = Encoder(latent_dim=512, n_segment=4)
    dummy_x = torch.randn(8, 4, 84, 84)
    z = enc(dummy_x)
    print("Encoder output shape:", z.shape)  # expect (8, 512)