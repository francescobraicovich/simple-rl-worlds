import torch
import torch.nn as nn
# Removed: from einops.layers.torch import Rearrange (will be in JEPAStateDecoder)
from .encoder import Encoder # Updated import
from .mlp import PredictorMLP # MLPEncoder is no longer directly needed here
from .decoder import StateDecoder # Import JEPAStateDecoder
from src.utils.weight_init import initialize_weights, print_num_parameters

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
                action_type: str, # New: 'discrete' or 'continuous'
                latent_dim,  # Output dim of encoder

                # Internal Predictor config (mirrors JEPA's predictor structure)
                predictor_hidden_dims,
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
        self.action_type = action_type # Store action_type
        # output_image_size is now directly passed and used by JEPAStateDecoder
        # self.input_channels = input_channels # Stored by encoder if needed
        # self.output_channels = output_channels # Stored by JEPAStateDecoder

        # Encoder instantiation
        self.encoder = Encoder(
            encoder_type=encoder_type,
            image_size=self._image_size_tuple,
            patch_size=patch_size,
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

        # JEPA-style predictor MLP (mimicking JEPA.predictor structure)
        predictor_input_actual_dim = latent_dim + action_emb_dim
            
        self.predictor = PredictorMLP(
            input_dim=predictor_input_actual_dim,
            hidden_dims=predictor_hidden_dims,  # Two hidden layers
            latent_dim=predictor_output_dim,  # Output dimension of the predictor
            activation_fn_str='gelu',  # JEPA uses GELU
            use_batch_norm=False,  # JEPA does not use batch norm in predictor
            dropout_rate=predictor_dropout_rate
        )

        # Instantiate JEPAStateDecoder
        _decoder_patch_size = jepa_decoder_patch_size if jepa_decoder_patch_size is not None else patch_size
        _output_image_size_tuple = output_image_size if isinstance(output_image_size, tuple) else (output_image_size, output_image_size)



        self.decoder = StateDecoder(
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
        print_num_parameters(self)


    def forward(self, current_state_img, action):
        """
        Args:
            current_state_img: (batch, channels, height, width)
            action: (batch, action_dim) for continuous, or (batch,) or (batch,1) for discrete
        Returns:
            predicted_next_state_img: (batch, output_channels, output_image_h, output_image_w)
        """
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
            if action.dtype != torch.float32: # Or match the model's default dtype
                action = action.float()
            embedded_action = self.action_embedding(action)  # (b, action_emb_dim)
        else:
            # This case should have been caught in __init__, but as a safeguard:
            raise ValueError(f"Unsupported action_type in forward pass: {self.action_type}")

        # 3. Concatenate and pass through predictor
        predictor_input = torch.cat((latent_s_t, embedded_action), dim=-1)
        # (b, latent_dim + action_emb_dim)
        predictor_output = self.predictor(predictor_input)  # (b, predictor_output_dim)

        # 4. Pass predictor's output to the JEPAStateDecoder instance
        # predictor_output is (b, input_latent_dim for JEPAStateDecoder)
        predicted_next_state_img = self.decoder(predictor_output)
        # Output: (b, output_channels, output_image_h, output_image_w)

        return predicted_next_state_img