import torch
import torch.nn as nn
import copy  # For deepcopying encoder for target network

# Import available encoders
from .vit import ViT
from .cnn import CNNEncoder
from .mlp import MLPEncoder
from src.utils.weight_init import initialize_weights


class JEPA(nn.Module):
    def __init__(self,
                 image_size,  # int or tuple (h,w)
                 patch_size,  # Primarily for ViT
                 input_channels,
                 action_dim,
                 action_emb_dim,
                 latent_dim,  # Output dim of any encoder
                 predictor_hidden_dim,
                 predictor_output_dim,
                 ema_decay=0.996,
                 encoder_type='vit',  # New: 'vit', 'cnn', 'mlp'
                 encoder_params: dict = None,  # New: dict to hold encoder-specific params
                 target_encoder_mode: str = "default" # Added: "default", "vjepa2", "none"
                 ):
        super().__init__()

        self.ema_decay = ema_decay
        self.target_encoder_mode = target_encoder_mode
        self._image_size_tuple = image_size if isinstance(
            image_size, tuple) else (image_size, image_size)

        # Encoder Instantiation (Online Encoder)
        if encoder_params is None:
            encoder_params = {}

        # Common parameters for all encoders that can be passed directly
        # For ViT, 'channels' is used for input_channels and 'dim' for latent_dim.
        # For CNN/MLP, 'input_channels' and 'latent_dim' are used directly.

        if encoder_type == 'vit':
            # ViT has specific parameter names ('channels' for input_channels, 'dim' for latent_dim)
            vit_constructor_params = {
                'image_size': self._image_size_tuple,
                'patch_size': patch_size,
                'channels': input_channels,  # ViT expects 'channels'
                'num_classes': 0,           # For feature extraction, not classification
                'dim': latent_dim,          # ViT expects 'dim' for the latent dimension
                'depth': encoder_params.get('depth', 6),
                'heads': encoder_params.get('heads', 8),
                'mlp_dim': encoder_params.get('mlp_dim', 1024),
                # Ensures (batch, dim) output
                'pool': encoder_params.get('pool', 'cls'),
                'dropout': encoder_params.get('dropout', 0.),
                'emb_dropout': encoder_params.get('emb_dropout', 0.)
            }
            self.online_encoder = ViT(**vit_constructor_params)
        elif encoder_type == 'cnn':
            cnn_constructor_params = {
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
            self.online_encoder = CNNEncoder(**cnn_constructor_params)
        elif encoder_type == 'mlp':
            mlp_constructor_params = {
                'input_channels': input_channels,
                'image_size': self._image_size_tuple,
                'latent_dim': latent_dim,
                'num_hidden_layers': encoder_params.get('num_hidden_layers', 2),
                'hidden_dim': encoder_params.get('hidden_dim', 512),
                'activation_fn_str': encoder_params.get('activation_fn_str', 'relu')
            }
            self.online_encoder = MLPEncoder(**mlp_constructor_params)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Target Encoder - initialized based on target_encoder_mode
        if self.target_encoder_mode == "none":
            self.target_encoder = None
        else:
            self.target_encoder = self._create_target_encoder()
            self._copy_weights_to_target_encoder()  # Initial copy
            for param in self.target_encoder.parameters():
                param.requires_grad = False

        # Action embedding
        self.action_embedding = nn.Linear(action_dim, action_emb_dim)

        # Predictor Network (MLP)
        # Input to predictor: latent_dim (from target_encoder(s_t)) + action_emb_dim
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_output_dim)
        )

        assert predictor_output_dim == latent_dim, \
            "Predictor output dimension must match encoder latent dimension for JEPA loss."

        self.apply(initialize_weights)

    def _create_target_encoder(self):
        return copy.deepcopy(self.online_encoder)

    @torch.no_grad()
    def _copy_weights_to_target_encoder(self):
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)

    @torch.no_grad()
    def _update_target_encoder_ema(self):
        if self.target_encoder is not None and self.target_encoder_mode != "none":
            for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                param_target.data = param_target.data * self.ema_decay + \
                    param_online.data * (1. - self.ema_decay)

    def forward(self, s_t, action, s_t_plus_1):
        # s_t: current state image (batch, c, h, w)
        # action: action taken (batch, action_dim)
        # s_t_plus_1: next state image (batch, c, h, w)

        embedded_action = self.action_embedding(action)  # a_t_emb

        if self.target_encoder_mode == "default":
            # Generate representations using TARGET encoder (EMA updated, no gradients for these ops)
            with torch.no_grad():
                target_encoded_s_t = self.target_encoder(s_t).detach() # z_t'
                target_for_predictor = self.target_encoder(s_t_plus_1).detach()  # z_{t+1}' (prediction target)

            # Predict the target representation of s_t+1 using target_encoded_s_t and action
            predictor_input = torch.cat((target_encoded_s_t, embedded_action), dim=-1)
            predicted_s_t_plus_1_embedding = self.predictor(predictor_input)  # \hat{z}_{t+1}

            # Generate representations using ONLINE encoder for auxiliary losses
            online_s_t_representation = self.online_encoder(s_t)                 # z_t
            online_s_t_plus_1_representation = self.online_encoder(s_t_plus_1)   # z_{t+1}

        elif self.target_encoder_mode == "vjepa2":
            online_s_t_representation = self.online_encoder(s_t) # z_t

            # EMA update for VJEPA2 happens here, before target encoder is used for s_t_plus_1
            self._update_target_encoder_ema()

            # Target for predictor is from target_encoder(s_t_plus_1)
            with torch.no_grad():
                if self.target_encoder is None: # Should not happen in vjepa2 mode due to __init__ logic
                    raise ValueError("Target encoder is None in vjepa2 mode, which is unexpected.")
                target_for_predictor = self.target_encoder(s_t_plus_1).detach() # z'_{t+1}

            # Predictor uses online_s_t_representation and action
            predictor_input = torch.cat((online_s_t_representation, embedded_action), dim=-1)
            predicted_s_t_plus_1_embedding = self.predictor(predictor_input) # \hat{z}_{t+1}

            # For vjepa2, only online_s_t_representation is needed for auxiliary loss.
            # online_s_t_plus_1_representation is not used/returned for this mode's specific auxiliary loss logic.
            online_s_t_plus_1_representation = None

        elif self.target_encoder_mode == "none":
            online_s_t_representation = self.online_encoder(s_t) # z_t
            online_s_t_plus_1_representation = self.online_encoder(s_t_plus_1) # z_{t+1}

            # Predictor uses online_s_t_representation and action
            predictor_input = torch.cat((online_s_t_representation, embedded_action), dim=-1)
            predicted_s_t_plus_1_embedding = self.predictor(predictor_input) # \hat{z}_{t+1}

            # Target for predictor is online_s_t_plus_1_representation (detached)
            target_for_predictor = online_s_t_plus_1_representation.detach()

        else:
            raise ValueError(f"Unsupported target_encoder_mode: {self.target_encoder_mode}")

        return predicted_s_t_plus_1_embedding, target_for_predictor, online_s_t_representation, online_s_t_plus_1_representation

    def perform_ema_update(self): # Renamed from update_target_network
        # This method is called by the training loop.
        # For "vjepa2", EMA is handled within the forward pass.
        # For "none", EMA is not used.
        if self.target_encoder_mode == "default":
            self._update_target_encoder_ema()
