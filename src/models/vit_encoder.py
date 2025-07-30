from torchvision.models.vision_transformer import VisionTransformer
import torch
import torch.nn as nn

class Encoder(VisionTransformer):
    def __init__(self, 
                image_size: int = 84,
                patch_size: int = 7,
                num_layers: int = 6,
                num_heads: int = 8,
                hidden_dim: int = 256,
                mlp_dim: int = 1024,
                in_chans: int = 4,  # input channels = n_segment (temporal frames)
                latent_dim: int = 512):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=0,  # 0 if you just want features
            in_chans=in_chans  # input channels = n_segment (temporal frames)
        )
        self.fc_enc = nn.Linear(self.embed_dim, latent_dim)
    
    def forward(self, x):
        x = super().forward(x)
        z = self.fc_enc(x)
        return z



if __name__ == "__main__":
    import inspect
    from torchvision.models.vision_transformer import VisionTransformer
    print(inspect.signature(VisionTransformer.__init__))
    enc = Encoder(latent_dim=512, in_chans=4)
    dummy_x = torch.randn(8, 4, 84, 84)
    z = enc(dummy_x)
    print("Encoder output shape:", z.shape)  # expect (8, 512)