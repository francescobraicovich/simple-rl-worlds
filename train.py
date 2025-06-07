import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gymnasium as gym
import torch.nn.functional as F

# ExperienceDataset removed
from utils.data_utils import collect_random_episodes
from models.encoder_decoder import StandardEncoderDecoder
from models.jepa import JEPA
from losses import VICRegLoss, BarlowTwinsLoss, DINOLoss


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_env_details(env_name):
    temp_env = gym.make(env_name)
    action_space = temp_env.action_space
    observation_space = temp_env.observation_space

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        action_type = 'discrete'
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
        action_type = 'continuous'
    else:
        temp_env.close()
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

    temp_env.close()
    print(f"Environment: {env_name}")
    print(f"Action space type: {action_type}, Action dimension: {action_dim}")
    print("Raw observation space:", observation_space)
    return action_dim, action_type, observation_space


def main():
    config = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    action_dim, action_type, _ = get_env_details(config['environment_name'])

    input_channels = config.get('input_channels', 3)
    image_h_w = config['image_size']

    # Encoder configuration
    encoder_type = config.get('encoder_type', 'vit')
    all_encoder_params_from_config = config.get('encoder_params', {})
    # Select the sub-dictionary for the chosen encoder_type, ensuring it's a dict
    specific_encoder_params = all_encoder_params_from_config.get(encoder_type, {})
    # Handles case where key exists but value is null
    if specific_encoder_params is None:
        specific_encoder_params = {}

    # Used by ViT and default for decoder
    global_patch_size = config.get('patch_size', 16)

    # If ViT is the encoder_type, ensure 'patch_size' is set in its specific_encoder_params.
    # ViT: 'patch_size' drct.
    if encoder_type == 'vit':
        if 'patch_size' not in specific_encoder_params or \
           specific_encoder_params['patch_size'] is None:
            specific_encoder_params['patch_size'] = global_patch_size

        # Set other ViT parameters with defaults if not provided in config's vit section.
        # ViT __init__ arguments.
        # The models themselves also have defaults, but setting here allows config override of those.
        vit_params_config = config.get('encoder_params', {}).get('vit', {})
        specific_encoder_params.setdefault(
            'depth', vit_params_config.get('depth', 6)
        )
        specific_encoder_params.setdefault(
            'heads', vit_params_config.get('heads', 8)
        )
        specific_encoder_params.setdefault(
            'mlp_dim', vit_params_config.get('mlp_dim', 1024)
        )
        specific_encoder_params.setdefault(
            'pool', vit_params_config.get('pool', 'cls')
        )
        specific_encoder_params.setdefault(
            'dropout', vit_params_config.get('dropout', 0.0)
        )
        specific_encoder_params.setdefault(
            'emb_dropout',
            vit_params_config.get('emb_dropout', 0.0)
        )


    print("Starting data collection...")
    key_max_steps = 'max_steps_per_episode_data_collection'
    dataset = collect_random_episodes(
        env_name=config['environment_name'],
        num_episodes=config.get('num_episodes_data_collection', 50),
        max_steps_per_episode=config.get(key_max_steps, 200),
        image_size=(image_h_w, image_h_w)
    )

    if len(dataset) == 0:
        print("No data collected. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=config.get('num_workers', 4), pin_memory=True)

    print(
        f"Initializing Standard Encoder-Decoder Model with "
        f"{encoder_type.upper()} encoder..."
    )
    std_enc_dec = StandardEncoderDecoder(
        image_size=image_h_w,
        # Passed for ViT encoder & default for decoder.
        # ViT gets patch_size from specific_encoder_params if 'vit'
        patch_size=global_patch_size,
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=config.get(
            'action_emb_dim',
            config.get('latent_dim', 128)
        ),
        latent_dim=config.get('latent_dim', 128),
        decoder_dim=config.get('decoder_dim', 128),
        decoder_depth=config.get(
            'decoder_depth', config.get('num_decoder_layers', 3)
        ),
        decoder_heads=config.get(
            'decoder_heads', config.get('num_heads', 6)
        ),
        decoder_mlp_dim=config.get(
            'decoder_mlp_dim', config.get('mlp_dim', 256)
        ),
        output_channels=input_channels,
        output_image_size=image_h_w,
        decoder_dropout=config.get('decoder_dropout', 0.0),
        encoder_type=encoder_type,
        # Pass the type-specific params
        encoder_params=specific_encoder_params,
        decoder_patch_size=config.get('decoder_patch_size', global_patch_size)
    ).to(device)
    optimizer_std_enc_dec = optim.AdamW(
        std_enc_dec.parameters(), lr=config.get('learning_rate', 0.0003)
    )
    mse_loss_fn = nn.MSELoss()

    print(f"Initializing JEPA Model with {encoder_type.upper()} encoder...")
    jepa_model = JEPA(
        image_size=image_h_w,
        # ViT encoder in JEPA gets patch_size from specific_encoder_params
        patch_size=global_patch_size,
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=config.get(
            'action_emb_dim', config.get('latent_dim', 128)),
        latent_dim=config.get('latent_dim', 128),
        predictor_hidden_dim=config.get('jepa_predictor_hidden_dim', 256),
        # Must match latent_dim
        predictor_output_dim=config.get('latent_dim', 128),
        ema_decay=config.get('ema_decay', 0.996),
        encoder_type=encoder_type,
        # Pass the type-specific params
        encoder_params=specific_encoder_params
    ).to(device)
    optimizer_jepa = optim.AdamW(
        jepa_model.parameters(),
        lr=config.get(
            'learning_rate_jepa', config.get('learning_rate', 0.0003)
        )
    )

    # --- New Auxiliary Loss Setup ---
    aux_loss_config = config.get('auxiliary_loss', {})
    aux_loss_type = aux_loss_config.get('type', 'vicreg').lower()
    aux_loss_weight = aux_loss_config.get('weight', 1.0)
    aux_loss_params_all = aux_loss_config.get('params', {})

    print(
        f"Initializing auxiliary loss: Type={aux_loss_type}, "
        f"Weight={aux_loss_weight}"
    )

    if aux_loss_type == 'vicreg':
        vicreg_params = aux_loss_params_all.get('vicreg', {})
        # Ensure default values if not in config, matching old behavior
        aux_loss_fn = VICRegLoss(
            sim_coeff=vicreg_params.get('sim_coeff', 0.0),
            std_coeff=vicreg_params.get('std_coeff', 25.0),
            cov_coeff=vicreg_params.get('cov_coeff', 1.0),
            eps=vicreg_params.get('eps', 1e-4)
        ).to(device)
        aux_loss_name = "VICReg"
    elif aux_loss_type == 'barlow_twins':
        bt_params = aux_loss_params_all.get('barlow_twins', {})
        aux_loss_fn = BarlowTwinsLoss(
            lambda_param=bt_params.get('lambda_param', 5e-3),
            eps=bt_params.get('eps', 1e-5),
            scale_loss=bt_params.get('scale_loss', 1.0)
        ).to(device)
        aux_loss_name = "BarlowTwins"
    elif aux_loss_type == 'dino':
        dino_params = aux_loss_params_all.get('dino', {})
        # DINOLoss requires out_dim, which is the latent_dim of the JEPA model
        # jepa_model should be initialized before this section.
        model_latent_dim = jepa_model.latent_dim  # Or config.get('latent_dim')
        aux_loss_fn = DINOLoss(
            out_dim=model_latent_dim,
            center_ema_decay=dino_params.get('center_ema_decay', 0.9),
            # eps in DINOLoss is for consistency, not heavily used
            eps=dino_params.get('eps', 1e-5)
        ).to(device)
        aux_loss_name = "DINO"
    else:
        print(
            f"Warning: Unknown auxiliary loss type '{aux_loss_type}'. "
            "No auxiliary loss will be applied."
        )
        aux_loss_fn = None
        aux_loss_name = "None"
        # Ensure no impact if not configured
        aux_loss_weight = 0

    # --- End New Auxiliary Loss Setup ---

    print(f"Starting training for {config.get('num_epochs', 10)} epochs...")
    for epoch in range(config.get('num_epochs', 10)):
        epoch_loss_std = 0
        epoch_loss_jepa_pred = 0
        epoch_loss_jepa_aux = 0

        num_batches = len(dataloader)
        if num_batches == 0:
            print(f"Epoch {epoch+1} has no data. Skipping.")
            continue

        for batch_idx, (s_t, a_t, s_t_plus_1) in enumerate(dataloader):
            s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)

            if action_type == 'discrete':
                if a_t.ndim == 1: a_t = a_t.unsqueeze(1) # Ensure (batch, 1)
                a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
            else:
                a_t_processed = a_t.float().to(device)

            # Standard Encoder-Decoder Training
            optimizer_std_enc_dec.zero_grad()
            predicted_s_t_plus_1 = std_enc_dec(s_t, a_t_processed)
            loss_std = mse_loss_fn(predicted_s_t_plus_1, s_t_plus_1)
            loss_std.backward()
            optimizer_std_enc_dec.step()
            epoch_loss_std += loss_std.item()

            # JEPA Training
            optimizer_jepa.zero_grad()
            pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = jepa_model(s_t, a_t_processed, s_t_plus_1)

            loss_jepa_pred = mse_loss_fn(pred_emb, target_emb_detached)

            if aux_loss_fn is not None and aux_loss_weight > 0:
                # Set aux_loss_fn to train mode if it has state (e.g. DINO's center)
                if hasattr(aux_loss_fn, 'train'):
                    aux_loss_fn.train()
                aux_term_s_t, _, _ = aux_loss_fn.calculate_reg_terms(
                    online_s_t_emb
                )
                aux_term_s_t_plus_1, _, _ = aux_loss_fn.calculate_reg_terms(
                    online_s_t_plus_1_emb
                )
                current_loss_jepa_aux = (
                    aux_term_s_t + aux_term_s_t_plus_1
                ) * 0.5
            else:
                current_loss_jepa_aux = torch.tensor(0.0, device=device)

            total_loss_jepa = loss_jepa_pred + current_loss_jepa_aux * aux_loss_weight
            total_loss_jepa.backward()
            optimizer_jepa.step()

            jepa_model.update_target_network()

            epoch_loss_jepa_pred += loss_jepa_pred.item()
            epoch_loss_jepa_aux += current_loss_jepa_aux.item()

            if (batch_idx + 1) % config.get('log_interval', 50) == 0:
                weighted_formatted_aux_loss = (
                    f"{(current_loss_jepa_aux * aux_loss_weight):.4f}"
                )
                log_parts = [
                    f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}:",
                    f"StdEncDec L: {loss_std.item():.4f} |",  # Shorter key
                    f"JEPA Pred L: {loss_jepa_pred.item():.4f},",  # Shorter key
                    f"{aux_loss_name} AuxRawL: {current_loss_jepa_aux.item():.4f}",  # Shorter key
                    f"(Weighted: {weighted_formatted_aux_loss}),",
                    f"Total JEPA L: {total_loss_jepa.item():.4f}"  # Shorter key
                ]
                print(" ".join(log_parts))

        avg_loss_std = epoch_loss_std / num_batches
        avg_loss_jepa_pred = epoch_loss_jepa_pred / num_batches
        avg_loss_jepa_aux_raw = (
            epoch_loss_jepa_aux / num_batches if num_batches > 0 else 0
        )

        print(f"Epoch {epoch+1}/{config.get('num_epochs', 10)} Summary:")
        print(f"  Avg StdEncDec L: {avg_loss_std:.4f}") # Shorter
        print(f"  Avg JEPA Pred L: {avg_loss_jepa_pred:.4f}") # Shorter
        print(
            f"  Avg JEPA {aux_loss_name} AuxRawL: " # Shorter
            f"{avg_loss_jepa_aux_raw:.4f}"
        )
        avg_total_jepa_loss = (
            avg_loss_jepa_pred + avg_loss_jepa_aux_raw * aux_loss_weight
        )
        print(f"  Avg Total JEPA L: {avg_total_jepa_loss:.4f}") # Shorter

    print("Training finished.")


if __name__ == '__main__':
    main()
