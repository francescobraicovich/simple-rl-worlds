# Contents for src/loss_setup.py
import torch
import torch.nn as nn
from src.losses.vicreg import VICRegLoss
from src.losses.barlow_twins import BarlowTwinsLoss
from src.losses.dino import DINOLoss # Assuming DINO loss is in src.losses.dino

def initialize_loss_functions(config, device, jepa_model_latent_dim=None):
    losses = {}

    losses['mse'] = nn.MSELoss()

    # Auxiliary Loss Setup
    aux_loss_config = config.get('models', {}).get('auxiliary_loss', {})
    aux_loss_type = aux_loss_config.get('type', 'vicreg').lower()
    aux_loss_weight = aux_loss_config.get('weight', 1.0) # Weight is used in training loop
    aux_loss_params_all = aux_loss_config.get('params', {})

    print(f"Initializing auxiliary loss: Type={aux_loss_type}, Weight={aux_loss_weight}")

    aux_fn = None
    aux_name = "None"

    if aux_loss_type == 'vicreg':
        vicreg_params = aux_loss_params_all.get('vicreg', {})
        aux_fn = VICRegLoss(
            sim_coeff=vicreg_params.get('sim_coeff', 0.0), # Adjusted from 25.0, common default is 0 or 1
            std_coeff=vicreg_params.get('std_coeff', 25.0), # Adjusted from 25.0
            cov_coeff=vicreg_params.get('cov_coeff', 1.0),
            eps=vicreg_params.get('eps', 1e-4)
        ).to(device)
        aux_name = "VICReg"
    elif aux_loss_type == 'barlow_twins':
        bt_params = aux_loss_params_all.get('barlow_twins', {})
        aux_fn = BarlowTwinsLoss(
            lambda_param=bt_params.get('lambda_param', 5e-3),
            eps=bt_params.get('eps', 1e-5),
            scale_loss=bt_params.get('scale_loss', 1.0)
        ).to(device)
        aux_name = "BarlowTwins"
    elif aux_loss_type == 'dino':
        dino_params = aux_loss_params_all.get('dino', {})
        if jepa_model_latent_dim is None:
            # Try to get from config as a fallback, though passing it is preferred
            jepa_model_latent_dim = config.get('models', {}).get('shared_latent_dim')
            if jepa_model_latent_dim is None:
                raise ValueError("jepa_model_latent_dim must be provided for DINO loss, or 'models.shared_latent_dim' in config.")

        aux_fn = DINOLoss(
            out_dim=jepa_model_latent_dim,
            center_ema_decay=dino_params.get('center_ema_decay', 0.9),
            eps=dino_params.get('eps', 1e-5)
        ).to(device)
        aux_name = "DINO"
    else:
        print(f"Warning: Unknown auxiliary loss type '{aux_loss_type}'. No auxiliary loss will be applied.")
        # aux_loss_weight will ensure no impact if it's 0 or aux_fn is None

    losses['aux_fn'] = aux_fn
    losses['aux_name'] = aux_name
    losses['aux_weight'] = aux_loss_weight # Pass weight along, to be used in training loop

    return losses
