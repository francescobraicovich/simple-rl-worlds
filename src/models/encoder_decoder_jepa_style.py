import torch
import torch.nn as nn
# Removed: from einops.layers.torch import Rearrange (will be in JEPAStateDecoder)
from .vit import ViT
from .cnn import CNNEncoder
from .mlp import MLPEncoder
from .jepa_state_decoder import JEPAStateDecoder # Import JEPAStateDecoder
from src.utils.weight_init import initialize_weights, count_parameters

class EncoderDecoderJEPAStyle(nn.Module):
    """
    Encoder-Decoder model that uses a JEPA-style predictor MLP followed by a JEPAStateDecoder.
    - Encoder processes current state.
    - Action is embedded.
    - Concatenated state-action embedding is fed to a predictor MLP (same arch as JEPA.predictor).
    - The predictor's output is then decoded by an instance of JEPAStateDecoder.
    """
    def __init__(self,
                image_size,  # int or tuple (h, w)
                patch_size,  # For ViT encoder, and default for jepa_decoder_patch_size
                input_channels,
                action_dim,
                action_emb_dim,
                latent_dim,  # Output dim of encoder

                # Internal Predictor config (mirrors JEPA's predictor structure)
                predictor_hidden_dim,
                predictor_output_dim,  # This is input_latent_dim for the internal JEPAStateDecoder
                # Config for the internal JEPAStateDecoder instance
                jepa_decoder_dim,
                jepa_decoder_depth,
                jepa_decoder_heads,
                jepa_decoder_mlp_dim,
                output_channels,         # For the final output image
                output_image_size,     # For the final output image (h,w)
                                 
                # Encoder config
                encoder_type='vit',
                encoder_params: dict = None,
                jepa_decoder_dropout=0.0,
                predictor_dropout_rate=0.0,

                jepa_decoder_patch_size=None # If None, defaults to patch_size
                ):
        super().__init__()

        self._image_size_tuple = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        # output_image_size is now directly passed and used by JEPAStateDecoder
        # self.input_channels = input_channels # Stored by encoder if needed
        # self.output_channels = output_channels # Stored by JEPAStateDecoder

        # Encoder instantiation (same as before)
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

        # JEPA-style predictor MLP (mimicking JEPA.predictor structure)
        predictor_input_actual_dim = latent_dim + action_emb_dim
        predictor_layers = [
            nn.Linear(predictor_input_actual_dim, predictor_hidden_dim),
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

        # Instantiate JEPAStateDecoder
        _decoder_patch_size = jepa_decoder_patch_size if jepa_decoder_patch_size is not None else patch_size
        _output_image_size_tuple = output_image_size if isinstance(output_image_size, tuple) else (output_image_size, output_image_size)


        self.decoder = JEPAStateDecoder(
            input_latent_dim=predictor_output_dim, # Output of self.predictor
            decoder_dim=jepa_decoder_dim,
            decoder_depth=jepa_decoder_depth,
            decoder_heads=jepa_decoder_heads,
            decoder_mlp_dim=jepa_decoder_mlp_dim,
            output_channels=output_channels,
            output_image_size=_output_image_size_tuple,
            decoder_dropout=jepa_decoder_dropout,
            decoder_patch_size=_decoder_patch_size
        )

        self.apply(initialize_weights)
        print(f"EncoderDecoderJEPAStyle initialized with {count_parameters(self):,} parameters.")

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
        predictor_input = torch.cat((latent_s_t, embedded_action), dim=-1)
        # (b, latent_dim + action_emb_dim)
        predictor_output = self.predictor(predictor_input)  # (b, predictor_output_dim)

        # 4. Pass predictor's output to the JEPAStateDecoder instance
        # predictor_output is (b, input_latent_dim for JEPAStateDecoder)
        predicted_next_state_img = self.decoder(predictor_output)
        # Output: (b, output_channels, output_image_h, output_image_w)

        return predicted_next_state_img