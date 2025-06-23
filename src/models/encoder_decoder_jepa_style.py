import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .vit import ViT
from .cnn import CNNEncoder
from .mlp import MLPEncoder
from src.utils.weight_init import initialize_weights

class EncoderDecoderJEPAStyle(nn.Module):
    """
    Encoder-Decoder model with JEPA-style predictor MLP and Transformer decoder.
    This model is designed for apples-to-apples comparison with JEPA:
    - Uses the same encoder options (ViT, CNN, MLP).
    - After encoding the current state, concatenates the action embedding.
    - Passes the concatenated vector through a JEPA-style predictor MLP.
    - The predictor's output is fed into a Transformer-based decoder to reconstruct the next state image in pixel space.
    - All architectural parameters are fully configurable for fair comparison.
    """
    def __init__(self,
                 image_size,  # int or tuple (h, w)
                 patch_size,  # For ViT and decoder's output patch structure
                 input_channels,
                 action_dim,
                 action_emb_dim,
                 latent_dim,  # Output dim of encoder
                 predictor_hidden_dim,
                 predictor_output_dim,  # Should match decoder input dim
                 predictor_dropout_rate=0.0,
                 decoder_dim=128,
                 decoder_depth=3,
                 decoder_heads=4,
                 decoder_mlp_dim=256,
                 output_channels=3,
                 output_image_size=None,  # int or tuple (h,w)
                 decoder_dropout=0.0,
                 encoder_type='vit',
                 encoder_params: dict = None,
                 decoder_patch_size: int = None):
        super().__init__()

        self._image_size_tuple = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self._output_image_size_tuple = output_image_size if output_image_size is not None else self._image_size_tuple
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Determine decoder_patch_size
        self.decoder_patch_size = decoder_patch_size if decoder_patch_size is not None else patch_size
        if self._output_image_size_tuple[0] % self.decoder_patch_size != 0 or \
           self._output_image_size_tuple[1] % self.decoder_patch_size != 0:
            raise ValueError(
                f"Output image dimensions ({self._output_image_size_tuple}) must be divisible by the decoder_patch_size ({self.decoder_patch_size}).")
        self.output_num_patches_h = self._output_image_size_tuple[0] // self.decoder_patch_size
        self.output_num_patches_w = self._output_image_size_tuple[1] // self.decoder_patch_size
        self.num_output_patches = self.output_num_patches_h * self.output_num_patches_w

        # Encoder instantiation (reuse logic from StandardEncoderDecoder)
        if encoder_params is None:
            encoder_params = {}
        if encoder_type == 'vit':
            vit_params = {
                'image_size': self._image_size_tuple,
                'patch_size': patch_size,
                'channels': input_channels,
                'num_classes': 0,
                'dim': latent_dim,
                'depth': encoder_params.get('depth', 6),
                'heads': encoder_params.get('heads', 8),
                'mlp_dim': encoder_params.get('mlp_dim', 1024),
                'pool': encoder_params.get('pool', 'cls'),
                'dropout': encoder_params.get('dropout', 0.),
                'emb_dropout': encoder_params.get('emb_dropout', 0.)
            }
            self.encoder = ViT(**vit_params)
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
            self.encoder = CNNEncoder(**cnn_params)
        elif encoder_type == 'mlp':
            mlp_params = {
                'input_channels': input_channels,
                'image_size': self._image_size_tuple,
                'latent_dim': latent_dim,
                'num_hidden_layers': encoder_params.get('num_hidden_layers', 2),
                'hidden_dim': encoder_params.get('hidden_dim', 512),
                'activation_fn_str': encoder_params.get('activation_fn_str', 'relu'),
                'dropout_rate': encoder_params.get('dropout_rate', 0.0)
            }
            self.encoder = MLPEncoder(**mlp_params)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Action embedding
        self.action_embedding = nn.Linear(action_dim, action_emb_dim)

        # JEPA-style predictor MLP
        predictor_layers = [
            nn.Linear(latent_dim + action_emb_dim, predictor_hidden_dim),
            nn.GELU()
        ]
        if predictor_dropout_rate > 0:
            predictor_layers.append(nn.Dropout(predictor_dropout_rate))
        predictor_layers.extend([
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim),
            nn.GELU()
        ])
        if predictor_dropout_rate > 0:
            predictor_layers.append(nn.Dropout(predictor_dropout_rate))
        predictor_layers.append(nn.Linear(predictor_hidden_dim, predictor_output_dim))
        self.predictor = nn.Sequential(*predictor_layers)

        # Transformer Decoder (same as StandardEncoderDecoder/JEPAStateDecoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_mlp_dim,
            dropout=decoder_dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=decoder_depth
        )
        self.decoder_query_tokens = nn.Parameter(torch.randn(1, self.num_output_patches, decoder_dim) * 0.02)
        output_patch_dim = self.output_channels * self.decoder_patch_size * self.decoder_patch_size
        self.to_pixels = nn.Linear(decoder_dim, output_patch_dim)
        self.patch_to_image = Rearrange(
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            p1=self.decoder_patch_size, p2=self.decoder_patch_size,
            h=self.output_num_patches_h, w=self.output_num_patches_w,
            c=self.output_channels
        )
        self.apply(initialize_weights)

    def forward(self, current_state_img, action):
        """
        Args:
            current_state_img: (batch, channels, height, width)
            action: (batch, action_dim)
        Returns:
            predicted_next_state_img: (batch, output_channels, output_image_h, output_image_w)
        """
        # 1. Encode current state
        latent_s_t = self.encoder(current_state_img)  # (b, latent_dim)
        # 2. Embed action
        embedded_action = self.action_embedding(action)  # (b, action_emb_dim)
        # 3. Concatenate and pass through predictor
        predictor_input = torch.cat((latent_s_t, embedded_action), dim=-1)  # (b, latent_dim + action_emb_dim)
        predictor_output = self.predictor(predictor_input)  # (b, predictor_output_dim)
        # 4. Project predictor output to decoder_dim and unsqueeze for memory
        decoder_memory = predictor_output.unsqueeze(1)  # (b, 1, decoder_dim) if dims match
        # 5. Prepare query tokens
        batch_size = current_state_img.shape[0]
        query_tokens = self.decoder_query_tokens.repeat(batch_size, 1, 1)  # (b, num_output_patches, decoder_dim)
        # 6. Pass through Transformer Decoder
        decoded_representation = self.transformer_decoder(tgt=query_tokens, memory=decoder_memory)
        # 7. Map to pixel values
        pixel_patches = self.to_pixels(decoded_representation)
        # 8. Reshape patches to image
        predicted_next_state_img = self.patch_to_image(pixel_patches)
        return predicted_next_state_img 