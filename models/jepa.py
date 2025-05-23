import torch
import torch.nn as nn
import copy # For deepcopying encoder for target network

from .vit import ViT # Assuming vit.py is in the same directory

class JEPA(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 input_channels, # Number of channels in the input image
                 action_dim,     # Dimension of the action space
                 action_emb_dim, # Dimension to embed actions into
                 latent_dim,     # Dimension of ViT output (encoder_output_dim)
                 predictor_hidden_dim, # Hidden dimension for the predictor MLP
                 predictor_output_dim, # Output dim of predictor, should match latent_dim
                 vit_depth,
                 vit_heads,
                 vit_mlp_dim,
                 vit_pool='cls',
                 vit_dropout=0.,
                 vit_emb_dropout=0.,
                 ema_decay=0.996):
        super().__init__()

        self.ema_decay = ema_decay

        # Online Encoder (ViT)
        self.online_encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            channels=input_channels,
            num_classes=0, # We want the latent representation
            dim=latent_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            pool=vit_pool,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout
        )

        # Target Encoder (ViT) - initialized as a copy of online, but non-trainable
        self.target_encoder = self._create_target_encoder()
        self._copy_weights_to_target_encoder() # Initial copy
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Action embedding
        self.action_embedding = nn.Linear(action_dim, action_emb_dim)

        # Predictor Network (MLP)
        # Takes target_encoded s_t and embedded action, predicts online_encoded s_t+1
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_output_dim) # Output should match latent_dim
        )
        
        # Ensure predictor_output_dim matches latent_dim for direct comparison
        assert predictor_output_dim == latent_dim,             "Predictor output dimension must match ViT latent dimension for JEPA loss."

    def _create_target_encoder(self):
        # Creates a deep copy of the online encoder
        return copy.deepcopy(self.online_encoder)

    @torch.no_grad()
    def _copy_weights_to_target_encoder(self):
        # Full copy of weights from online to target
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)

    @torch.no_grad()
    def _update_target_encoder_ema(self):
        # EMA update for target encoder parameters
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = param_target.data * self.ema_decay + param_online.data * (1. - self.ema_decay)

    def forward(self, s_t, action, s_t_plus_1):
        # s_t: current state image (batch, c, h, w)
        # action: action taken (batch, action_dim)
        # s_t_plus_1: next state image (batch, c, h, w)

        # Encode s_t with the target encoder (no gradients)
        with torch.no_grad():
            target_encoded_s_t = self.target_encoder(s_t).detach() # (b, latent_dim)

        # Encode s_t_plus_1 with the online encoder (gradients will flow here for VICReg)
        # This is the "y" in I-JEPA paper, or the target for the predictor's output.
        online_encoded_s_t_plus_1 = self.online_encoder(s_t_plus_1) # (b, latent_dim)

        # Embed action
        embedded_action = self.action_embedding(action) # (b, action_emb_dim)

        # Predict the embedding of s_t+1
        # Input to predictor: target_encoded_s_t and embedded_action
        predictor_input = torch.cat((target_encoded_s_t, embedded_action), dim=-1)
        predicted_s_t_plus_1_embedding = self.predictor(predictor_input) # (b, latent_dim)

        # The JEPA loss will be calculated between:
        # predicted_s_t_plus_1_embedding (output of predictor)
        # and online_encoded_s_t_plus_1.detach() (target for prediction, stop gradient to it from predictor loss)
        
        # For VICReg, we also need embeddings from the online encoder.
        # Specifically, we might apply VICReg to online_encoded_s_t and/or online_encoded_s_t_plus_1.
        # Let's also get online_encoded_s_t for potential use in VICReg.
        online_encoded_s_t = self.online_encoder(s_t) # (b, latent_dim)

        # Return values needed for loss calculation:
        # - predicted_s_t_plus_1_embedding
        # - online_encoded_s_t_plus_1 (this is the target for the prediction loss)
        # - online_encoded_s_t (for VICReg)
        # - online_encoded_s_t_plus_1 (again, for VICReg - can be the same tensor as above)
        
        return predicted_s_t_plus_1_embedding, online_encoded_s_t_plus_1.detach(), online_encoded_s_t, online_encoded_s_t_plus_1


    def update_target_network(self):
        # Wrapper for EMA update
        self._update_target_encoder_ema()
