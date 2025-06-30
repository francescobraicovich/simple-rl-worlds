import torch
import torch.nn as nn
from einops.layers.torch import Rearrange  # For use in __init__ as a layer

# Import available encoders
from .encoder import Encoder # Updated import
from src.utils.weight_init import initialize_weights, print_num_parameters


class StandardEncoderDecoder(nn.Module):
    def __init__(self,
                 image_size,  # int or tuple (h, w)
                 patch_size,  # Primarily for ViT and decoder's output patch structure
                 input_channels,
                 action_dim,
                 action_type, # Added action_type
                 action_emb_dim,  # Added missing action_emb_dim parameter
                 latent_dim,  # This is the output dim of any encoder
                 decoder_dim,
                 decoder_depth,
                 decoder_heads,
                 decoder_mlp_dim,
                 output_channels,
                 output_image_size,  # int or tuple (h,w)
                 decoder_dropout=0.,
                 encoder_type='vit',  # New: 'vit', 'cnn', 'mlp'
                 encoder_params: dict = None,  # New: dict to hold encoder-specific params
                 decoder_patch_size: int = None):  # New: explicit patch size for decoder output
        super().__init__()

        self._image_size_tuple = image_size
        self._output_image_size_tuple = output_image_size

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.action_type = action_type  # Store action_type

        # Determine decoder_patch_size
        # If not provided, default to 'patch_size' (which was historically ViT's patch_size)
        self.decoder_patch_size = decoder_patch_size if decoder_patch_size is not None else patch_size

        if self._output_image_size_tuple[0] % self.decoder_patch_size != 0 or \
           self._output_image_size_tuple[1] % self.decoder_patch_size != 0:
            raise ValueError(
                f"Output image dimensions ({self._output_image_size_tuple}) must be divisible by the decoder_patch_size ({self.decoder_patch_size}).")

        self.output_num_patches_h = self._output_image_size_tuple[0] // self.decoder_patch_size
        self.output_num_patches_w = self._output_image_size_tuple[1] // self.decoder_patch_size
        num_output_patches = self.output_num_patches_h * self.output_num_patches_w

        # Encoder Instantiation
        self.encoder = Encoder(
            encoder_type=encoder_type,
            image_size=self._image_size_tuple,
            patch_size=patch_size, # This is ViT's patch_size, other encoders might not use it directly
            input_channels=input_channels,
            latent_dim=latent_dim,
            encoder_params=encoder_params
        )

        # Action embedding
        if self.action_type == 'discrete':
            # action_dim is num_actions for discrete
            self.action_embedding = nn.Embedding(action_dim, action_emb_dim)
        elif self.action_type == 'continuous':
            # action_dim is the dimensionality of the action vector
            self.action_embedding = nn.Linear(action_dim, action_emb_dim)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        # Decoder input projection
        self.decoder_input_dim = latent_dim + action_emb_dim
        self.decoder_input_projection = nn.Linear(
            self.decoder_input_dim, decoder_dim)

        # Transformer Decoder
        decoder_layer_args = {
            'd_model': decoder_dim,
            'nhead': decoder_heads,
            'dim_feedforward': decoder_mlp_dim,
            'dropout': decoder_dropout,
            'batch_first': True
        }
        decoder_layer = nn.TransformerDecoderLayer(**decoder_layer_args)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_depth)

        self.decoder_query_tokens = nn.Parameter(
            torch.randn(1, num_output_patches, decoder_dim) * 0.02) # Standardized init

        # Final layer to project decoder output to patch pixel values
        output_patch_dim = self.output_channels * \
            self.decoder_patch_size * self.decoder_patch_size
        self.to_pixels = nn.Linear(decoder_dim, output_patch_dim)

        # Layer to reconstruct image from patches
        self.patch_to_image = Rearrange(
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', # Standardized Rearrange
            p1=self.decoder_patch_size, p2=self.decoder_patch_size,
            h=self.output_num_patches_h, w=self.output_num_patches_w,
            c=self.output_channels
        )
        self.apply(initialize_weights)
        print_num_parameters(self)

    def forward(self, current_state_img, action):
        # current_state_img: (b, c, h, w)
        # action: (batch, action_dim) for continuous, or (batch,) or (batch,1) for discrete

        # 1. Encode current state
        latent_s_t = self.encoder(current_state_img)  # (b, latent_dim)

        # 2. Embed action
        if self.action_type == 'discrete':
            # Ensure action is long and squeezed if it's (batch, 1)
            if action.ndim == 2 and action.shape[1] == 1:
                action = action.squeeze(1)
            if action.dtype != torch.long:
                action = action.long()
            embedded_action = self.action_embedding(action)  # (b, action_emb_dim)
        elif self.action_type == 'continuous':
            # Ensure action is float
            if action.dtype != torch.float32:  # Or match the model's default dtype
                action = action.float()
            embedded_action = self.action_embedding(action)  # (b, action_emb_dim)
        else:
            # This case should have been caught in __init__, but as a safeguard:
            raise ValueError(f"Unsupported action_type in forward pass: {self.action_type}")

        # 3. Combine latent state and action for decoder memory
        decoder_memory_input = torch.cat((latent_s_t, embedded_action), dim=-1)
        decoder_memory = self.decoder_input_projection(decoder_memory_input)
        # (b, 1, decoder_dim) - memory for the decoder
        decoder_memory = decoder_memory.unsqueeze(1)

        # 4. Prepare query tokens
        batch_size = current_state_img.shape[0]
        query_tokens = self.decoder_query_tokens.repeat(
            batch_size, 1, 1)  # (b, num_output_patches, decoder_dim)

        # 5. Pass through Transformer Decoder
        decoded_patches_representation = self.transformer_decoder(
            tgt=query_tokens, memory=decoder_memory)
        # Output: (b, num_output_patches, decoder_dim)

        # 6. Project to pixel values for each patch
        predicted_patches = self.to_pixels(decoded_patches_representation)
        # Output: (b, num_output_patches, output_patch_dim)

        # 7. Rearrange patches back into an image
        # predicted_patches_reshaped = rearrange(
        #     predicted_patches, 'b (ph pw) d -> (b ph pw) d', ph=self.output_num_patches_h, pw=self.output_num_patches_w)
        # predicted_s_t_plus_1 = self.patch_to_image(predicted_patches_reshaped)

        predicted_s_t_plus_1 = self.patch_to_image(predicted_patches) # Directly use (b, num_patches, patch_dim)
        # Output: (b, output_channels, output_image_size_h, output_image_size_w)

        return predicted_s_t_plus_1
