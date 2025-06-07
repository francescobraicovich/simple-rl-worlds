import torch
import torch.nn as nn
from einops import rearrange # For use in forward method
from einops.layers.torch import Rearrange # For use in __init__ as a layer

# Import available encoders
from .vit import ViT
from .cnn import CNNEncoder
from .mlp import MLPEncoder

class StandardEncoderDecoder(nn.Module):
    def __init__(self,
                 image_size, # int or tuple (h, w)
                 patch_size, # Primarily for ViT and decoder's output patch structure
                 input_channels,
                 action_dim,
                 action_emb_dim,
                 latent_dim, # This is the output dim of any encoder
                 decoder_dim,
                 decoder_depth,
                 decoder_heads,
                 decoder_mlp_dim,
                 output_channels,
                 output_image_size, # int or tuple (h,w)
                 decoder_dropout=0.,
                 encoder_type='vit', # New: 'vit', 'cnn', 'mlp'
                 encoder_params: dict = None, # New: dict to hold encoder-specific params
                 decoder_patch_size: int = None): # New: explicit patch size for decoder output
        super().__init__()

        self._image_size_tuple = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self._output_image_size_tuple = output_image_size if isinstance(output_image_size, tuple) else (output_image_size, output_image_size)

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Determine decoder_patch_size
        # If not provided, default to 'patch_size' (which was historically ViT's patch_size)
        self.decoder_patch_size = decoder_patch_size if decoder_patch_size is not None else patch_size

        if self._output_image_size_tuple[0] % self.decoder_patch_size != 0 or \
           self._output_image_size_tuple[1] % self.decoder_patch_size != 0:
            raise ValueError(f"Output image dimensions ({self._output_image_size_tuple}) must be divisible by the decoder_patch_size ({self.decoder_patch_size}).")

        self.output_num_patches_h = self._output_image_size_tuple[0] // self.decoder_patch_size
        self.output_num_patches_w = self._output_image_size_tuple[1] // self.decoder_patch_size
        num_output_patches = self.output_num_patches_h * self.output_num_patches_w

        # Encoder Instantiation
        if encoder_params is None:
            encoder_params = {}

        if encoder_type == 'vit':
            vit_params = {
                'image_size': self._image_size_tuple,
                'patch_size': patch_size, # ViT needs its specific patch_size for input processing
                'channels': input_channels,
                'num_classes': 0, # Not used for feature extraction
                'dim': latent_dim,
                'depth': encoder_params.get('depth', 6),
                'heads': encoder_params.get('heads', 8),
                'mlp_dim': encoder_params.get('mlp_dim', 1024),
                'pool': encoder_params.get('pool', 'cls'), # Make sure ViT returns (b, dim)
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
                'fc_hidden_dim': encoder_params.get('fc_hidden_dim', None)
            }
            self.encoder = CNNEncoder(**cnn_params)
        elif encoder_type == 'mlp':
            mlp_params = {
                'input_channels': input_channels,
                'image_size': self._image_size_tuple,
                'latent_dim': latent_dim,
                'num_hidden_layers': encoder_params.get('num_hidden_layers', 2),
                'hidden_dim': encoder_params.get('hidden_dim', 512),
                'activation_fn_str': encoder_params.get('activation_fn_str', 'relu')
            }
            self.encoder = MLPEncoder(**mlp_params)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Action embedding
        self.action_embedding = nn.Linear(action_dim, action_emb_dim)

        # Decoder input projection
        self.decoder_input_dim = latent_dim + action_emb_dim
        self.decoder_input_projection = nn.Linear(self.decoder_input_dim, decoder_dim)

        # Transformer Decoder
        decoder_layer_args = {
            'd_model': decoder_dim,
            'nhead': decoder_heads,
            'dim_feedforward': decoder_mlp_dim,
            'dropout': decoder_dropout,
            'batch_first': True
        }
        decoder_layer = nn.TransformerDecoderLayer(**decoder_layer_args)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        
        self.decoder_query_tokens = nn.Parameter(torch.randn(1, num_output_patches, decoder_dim))

        # Final layer to project decoder output to patch pixel values
        output_patch_dim = self.output_channels * self.decoder_patch_size * self.decoder_patch_size
        self.to_pixels = nn.Linear(decoder_dim, output_patch_dim)
        
        # Layer to reconstruct image from patches
        self.patch_to_image = Rearrange(
            '(b ph pw) (p1 p2 c) -> b c (ph p1) (pw p2)',
            p1=self.decoder_patch_size, p2=self.decoder_patch_size,
            ph=self.output_num_patches_h, pw=self.output_num_patches_w,
            c=self.output_channels
        )


    def forward(self, current_state_img, action):
        # current_state_img: (b, c, h, w)
        # action: (b, action_dim)

        # 1. Encode current state
        latent_s_t = self.encoder(current_state_img)  # (b, latent_dim)

        # 2. Embed action
        embedded_action = self.action_embedding(action) # (b, action_emb_dim)

        # 3. Combine latent state and action for decoder memory
        decoder_memory_input = torch.cat((latent_s_t, embedded_action), dim=-1)
        decoder_memory = self.decoder_input_projection(decoder_memory_input)
        decoder_memory = decoder_memory.unsqueeze(1) # (b, 1, decoder_dim) - memory for the decoder

        # 4. Prepare query tokens
        batch_size = current_state_img.shape[0]
        query_tokens = self.decoder_query_tokens.repeat(batch_size, 1, 1) # (b, num_output_patches, decoder_dim)

        # 5. Pass through Transformer Decoder
        decoded_patches_representation = self.transformer_decoder(tgt=query_tokens, memory=decoder_memory)
        # Output: (b, num_output_patches, decoder_dim)

        # 6. Project to pixel values for each patch
        predicted_patches = self.to_pixels(decoded_patches_representation)
        # Output: (b, num_output_patches, output_patch_dim)
        
        # 7. Rearrange patches back into an image
        predicted_patches_reshaped = rearrange(predicted_patches, 'b (ph pw) d -> (b ph pw) d', ph=self.output_num_patches_h, pw=self.output_num_patches_w)

        predicted_s_t_plus_1 = self.patch_to_image(predicted_patches_reshaped)
        # Output: (b, output_channels, output_image_size_h, output_image_size_w)

        return predicted_s_t_plus_1
