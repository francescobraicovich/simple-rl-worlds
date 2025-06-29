import torch
import torch.nn as nn
import copy  # For deepcopying encoder for target network

# Import available encoders
from .encoder import Encoder # Updated import
from .mlp import PredictorMLP # MLPEncoder is no longer directly needed here
from src.utils.weight_init import initialize_weights, count_parameters, print_num_parameters


class JEPA(nn.Module):
    def __init__(self,
                 image_size,  # int or tuple (h,w)
                 patch_size,  # Primarily for ViT
                 input_channels,
                 action_dim,
                 action_emb_dim,
                 action_type: str, # New: 'discrete' or 'continuous'
                 latent_dim,  # Output dim of any encoder
                 predictor_hidden_dims,
                 ema_decay=0.996,
                 encoder_type='vit',  # New: 'vit', 'cnn', 'mlp'
                 encoder_params: dict = None,  # New: dict to hold encoder-specific params
                 target_encoder_mode: str = "default", # Added: "default", "vjepa2", "none"
                 predictor_dropout_rate: float = 0.0  # Added
                 ):
        super().__init__()

        self.ema_decay = ema_decay
        self.target_encoder_mode = target_encoder_mode
        self.predictor_dropout_rate = predictor_dropout_rate  # Stored
        self.action_type = action_type # Store action_type
        self._image_size_tuple = image_size if isinstance(
            image_size, tuple) else (image_size, image_size)

        # Encoder Instantiation (Online Encoder)
        self.online_encoder = Encoder(
            encoder_type=encoder_type,
            image_size=self._image_size_tuple,
            patch_size=patch_size,
            input_channels=input_channels,
            latent_dim=latent_dim,
            encoder_params=encoder_params
        )

        # Target Encoder - initialized based on target_encoder_mode
        if self.target_encoder_mode == "none":
            self.target_encoder = None
        else:
            self.target_encoder = self._create_target_encoder()
            self._copy_weights_to_target_encoder()  # Initial copy
            for param in self.target_encoder.parameters():
                param.requires_grad = False

        # Action embedding
        if self.action_type == 'discrete':
            # action_dim is num_actions for discrete
            self.action_embedding = nn.Embedding(action_dim, action_emb_dim)
        elif self.action_type == 'continuous':
            # action_dim is the dimensionality of the action vector
            self.action_embedding = nn.Linear(action_dim, action_emb_dim)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        # Predictor Network (MLP)
        # JEPA-style predictor MLP (mimicking JEPA.predictor structure)
        predictor_input_actual_dim = latent_dim + action_emb_dim
            
        self.predictor = PredictorMLP(
            input_dim=predictor_input_actual_dim,
            hidden_dims=predictor_hidden_dims,  # Two hidden layers
            latent_dim=latent_dim,  # Output dim of predictor
            activation_fn_str='gelu',  # JEPA uses GELU
            use_batch_norm=False,  # JEPA does not use batch norm in predictor
            dropout_rate=predictor_dropout_rate
        )

        self.apply(initialize_weights)
        print_num_parameters(self)

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
        # action: action taken (batch, action_dim) for continuous, or (batch,) or (batch,1) for discrete
        # s_t_plus_1: next state image (batch, c, h, w)

        if self.action_type == 'discrete':
            # Ensure action is long and squeezed if it's (batch, 1)
            if action.ndim == 2 and action.shape[1] == 1:
                action = action.squeeze(1)
            if action.dtype != torch.long:
                action = action.long()
            embedded_action = self.action_embedding(action) # (batch, action_emb_dim)
        elif self.action_type == 'continuous':
            # Ensure action is float
            if action.dtype != torch.float32:
                action = action.float() # Or match the model's default dtype
            embedded_action = self.action_embedding(action)  # (batch, action_emb_dim)
        else:
            # This case should have been caught in __init__, but as a safeguard:
            raise ValueError(f"Unsupported action_type in forward pass: {self.action_type}")


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
        if self.target_encoder_mode in ["default", "vjepa2"]:
            self._update_target_encoder_ema()
