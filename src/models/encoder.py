import torch.nn as nn
from .vit import ViT
from .cnn import CNNEncoder
from .mlp import MLPEncoder

class Encoder(nn.Module):
    def __init__(self,
                 encoder_type: str,
                 image_size,  # int or tuple (h, w)
                 patch_size: int,  # Primarily for ViT
                 input_channels: int,
                 latent_dim: int,  # This is the output dim of any encoder
                 encoder_params: dict = None):
        super().__init__()

        self._image_size_tuple = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        if encoder_params is None:
            encoder_params = {}

        if encoder_type == 'vit':
            vit_params = {
                'image_size': self._image_size_tuple,
                'patch_size': patch_size,
                'channels': input_channels,
                'num_classes': 0,  # Not used for feature extraction
                'dim': latent_dim,
                'depth': encoder_params.get('depth', 6),
                'heads': encoder_params.get('heads', 8),
                'mlp_dim': encoder_params.get('mlp_dim', 1024),
                'pool': encoder_params.get('pool', 'cls'), # Ensures (batch, dim) output
                'dropout': encoder_params.get('dropout', 0.),
                'emb_dropout': encoder_params.get('emb_dropout', 0.)
            }
            self.encoder_model = ViT(**vit_params)
        elif encoder_type == 'cnn':
            cnn_params = {
                'input_channels': input_channels,
                'image_size': self._image_size_tuple,
                'latent_dim': latent_dim,
                'num_conv_layers': encoder_params.get('num_conv_layers', 3),
                'base_filters': encoder_params.get('base_filters', 32),
                'kernel_size': encoder_params.get('kernel_size', 3),
                'stride': encoder_params.get('stride', 2),
                'padding': encoder_params.get('padding', 1),
                'activation_fn_str': encoder_params.get('activation_fn_str', 'relu'),
                'fc_hidden_dim': encoder_params.get('fc_hidden_dim', None),
                'dropout_rate': encoder_params.get('dropout_rate', 0.0)
            }
            self.encoder_model = CNNEncoder(**cnn_params)
        elif encoder_type == 'mlp':
            mlp_params = {
                'input_channels': input_channels,
                'image_size': self._image_size_tuple, # MLPEncoder expects image_size to flatten
                'latent_dim': latent_dim,
                'num_hidden_layers': encoder_params.get('num_hidden_layers', 2),
                'hidden_dim': encoder_params.get('hidden_dim', 512),
                'activation_fn_str': encoder_params.get('activation_fn_str', 'relu'),
                'dropout_rate': encoder_params.get('dropout_rate', 0.0)
            }
            self.encoder_model = MLPEncoder(**mlp_params)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def forward(self, x):
        return self.encoder_model(x)
