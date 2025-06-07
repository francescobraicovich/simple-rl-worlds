import yaml
import os  # For path joining
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
from models.mlp import RewardPredictorMLP
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
        raise ValueError(
            f"Unsupported action space type: {type(action_space)}")

    temp_env.close()
    print(f"Environment: {env_name}")
    print(f"Action space type: {action_type}, Action dimension: {action_dim}")
    print("Raw observation space:", observation_space)
    return action_dim, action_type, observation_space


def main():
    config = load_config()

    # Load Reward Predictor Configurations
    reward_pred_config = config.get('reward_predictors', {})
    enc_dec_mlp_config = reward_pred_config.get(
        'encoder_decoder_reward_mlp', {})
    jepa_mlp_config = reward_pred_config.get('jepa_reward_mlp', {})

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    action_dim, action_type, _ = get_env_details(config['environment_name'])

    # Early Stopping Configuration
    early_stopping_config = config.get('early_stopping', {})
    patience = early_stopping_config.get('patience', 10)
    delta = early_stopping_config.get('delta', 0.001)
    # metric_enc_dec = early_stopping_config.get('metric_enc_dec', 'val_loss_enc_dec') # Using direct val_loss_std
    # metric_jepa = early_stopping_config.get('metric_jepa', 'val_total_loss_jepa') # Using direct avg_total_val_loss_jepa
    checkpoint_path_enc_dec = early_stopping_config.get(
        'checkpoint_path_enc_dec', 'best_encoder_decoder.pth')
    checkpoint_path_jepa = early_stopping_config.get(
        'checkpoint_path_jepa', 'best_jepa.pth')
    validation_split = early_stopping_config.get('validation_split', 0.2)
    print(
        f"Early stopping configuration: patience={patience}, delta={delta}, validation_split={validation_split}")
    print(
        f"  Enc/Dec checkpoint: {checkpoint_path_enc_dec}, JEPA checkpoint: {checkpoint_path_jepa}")

    input_channels = config.get('input_channels', 3)
    image_h_w = config['image_size']

    # Encoder configuration
    encoder_type = config.get('encoder_type', 'vit')
    all_encoder_params_from_config = config.get('encoder_params', {})
    # Select the sub-dictionary for the chosen encoder_type, ensuring it's a dict
    specific_encoder_params = all_encoder_params_from_config.get(
        encoder_type, {})
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
    # Pass image_size as a tuple directly as expected by the modified function
    img_size_tuple = (image_h_w, image_h_w)
    train_dataset, val_dataset = collect_random_episodes(
        env_name=config['environment_name'],
        num_episodes=config.get('num_episodes_data_collection', 50),
        max_steps_per_episode=config.get(key_max_steps, 200),
        image_size=img_size_tuple,  # Pass the tuple here
        validation_split_ratio=validation_split
    )

    if len(train_dataset) == 0:
        print("No training data collected. Exiting.")
        return

    if validation_split > 0 and len(val_dataset) == 0:
        print("Warning: Validation split is > 0 but no validation data was collected. Check dataset size and split ratio.")

    train_dataloader = DataLoader(train_dataset, batch_size=config.get(
        'batch_size', 32), shuffle=True, num_workers=config.get('num_workers', 4), pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.get(
        'batch_size', 32), shuffle=False, num_workers=config.get('num_workers', 4), pin_memory=True)

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

    # Initialize Reward MLPs and Optimizers
    reward_mlp_enc_dec = None
    optimizer_reward_mlp_enc_dec = None
    if enc_dec_mlp_config.get('enabled', False):
        print("Initializing Reward MLP for Encoder-Decoder...")
        if enc_dec_mlp_config.get('input_type') == "flatten":
            input_dim_enc_dec = input_channels * image_h_w * image_h_w
        else:
            # Defaulting to latent_dim if not flatten, or could raise error
            # For now, let's assume "flatten" is the primary supported mode from config for enc-dec
            print(
                f"Warning: encoder_decoder_reward_mlp input_type is '{enc_dec_mlp_config.get('input_type')}'. Defaulting to flattened image dim.")
            input_dim_enc_dec = input_channels * image_h_w * image_h_w

        reward_mlp_enc_dec = RewardPredictorMLP(
            input_dim=input_dim_enc_dec,
            hidden_dims=enc_dec_mlp_config.get('hidden_dims', [128, 64]),
            activation_fn_str=enc_dec_mlp_config.get('activation', 'relu'),
            use_batch_norm=enc_dec_mlp_config.get('use_batch_norm', False)
        ).to(device)
        optimizer_reward_mlp_enc_dec = optim.AdamW(
            reward_mlp_enc_dec.parameters(),
            lr=enc_dec_mlp_config.get('learning_rate', 0.0003)
        )
        print(f"Encoder-Decoder Reward MLP: {reward_mlp_enc_dec}")

    reward_mlp_jepa = None
    optimizer_reward_mlp_jepa = None
    if jepa_mlp_config.get('enabled', False):
        print("Initializing Reward MLP for JEPA...")
        # JEPA's reward MLP uses encoder's latent output
        input_dim_jepa = config.get('latent_dim', 128)
        reward_mlp_jepa = RewardPredictorMLP(
            input_dim=input_dim_jepa,
            hidden_dims=jepa_mlp_config.get('hidden_dims', [128, 64]),
            activation_fn_str=jepa_mlp_config.get('activation', 'relu'),
            use_batch_norm=jepa_mlp_config.get('use_batch_norm', False)
        ).to(device)
        optimizer_reward_mlp_jepa = optim.AdamW(
            reward_mlp_jepa.parameters(),
            lr=jepa_mlp_config.get('learning_rate', 0.0003)
        )
        print(f"JEPA Reward MLP: {reward_mlp_jepa}")

    print(f"Starting training for {config.get('num_epochs', 10)} epochs...")

    # Early Stopping Trackers
    best_val_loss_enc_dec = float('inf')
    epochs_no_improve_enc_dec = 0
    best_val_loss_jepa = float('inf')
    epochs_no_improve_jepa = 0
    early_stop_enc_dec = False
    early_stop_jepa = False
    # checkpoint_path_enc_dec and checkpoint_path_jepa are already loaded from config

    for epoch in range(config.get('num_epochs', 10)):
        epoch_loss_std = 0
        epoch_loss_jepa_pred = 0
        epoch_loss_jepa_aux = 0
        epoch_loss_jepa_aux = 0
        epoch_loss_reward_enc_dec = 0
        epoch_loss_reward_jepa = 0

        num_train_batches = len(train_dataloader)
        if num_train_batches == 0:
            print(f"Epoch {epoch+1} has no training data. Skipping.")
            continue

        # Main model training loop
        std_enc_dec.train()
        jepa_model.train()
        if hasattr(aux_loss_fn, 'train'):  # Set DINO loss to train mode
            aux_loss_fn.train()

        # === Training Phase ===
        if not early_stop_enc_dec or not early_stop_jepa:  # Only loop if at least one model is training
            for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
                s_t, r_t, s_t_plus_1 = s_t.to(device), r_t.to(
                    device).float().unsqueeze(1), s_t_plus_1.to(device)

                # Process actions
                if action_type == 'discrete':
                    if a_t.ndim == 1:
                        a_t = a_t.unsqueeze(1)
                    a_t_processed = F.one_hot(
                        a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_processed = a_t.float().to(device)

                # Standard Encoder-Decoder Training
                if not early_stop_enc_dec:
                    optimizer_std_enc_dec.zero_grad()
                    predicted_s_t_plus_1 = std_enc_dec(s_t, a_t_processed)
                    loss_std = mse_loss_fn(predicted_s_t_plus_1, s_t_plus_1)
                    loss_std.backward()
                    optimizer_std_enc_dec.step()
                    epoch_loss_std += loss_std.item()

                # JEPA Training
                if not early_stop_jepa:
                    optimizer_jepa.zero_grad()
                    pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = jepa_model(
                        s_t, a_t_processed, s_t_plus_1)
                    loss_jepa_pred = mse_loss_fn(pred_emb, target_emb_detached)

                    current_loss_jepa_aux = torch.tensor(0.0, device=device)
                    if aux_loss_fn is not None and aux_loss_weight > 0:
                        aux_term_s_t, _, _ = aux_loss_fn.calculate_reg_terms(
                            online_s_t_emb)
                        aux_term_s_t_plus_1, _, _ = aux_loss_fn.calculate_reg_terms(
                            online_s_t_plus_1_emb)
                        current_loss_jepa_aux = (
                            aux_term_s_t + aux_term_s_t_plus_1) * 0.5

                    total_loss_jepa = loss_jepa_pred + current_loss_jepa_aux * aux_loss_weight
                    total_loss_jepa.backward()
                    optimizer_jepa.step()
                    jepa_model.update_target_network()
                    epoch_loss_jepa_pred += loss_jepa_pred.item()
                    epoch_loss_jepa_aux += current_loss_jepa_aux.item()

                if (batch_idx + 1) % config.get('log_interval', 50) == 0:
                    log_msg = f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_train_batches}:"
                    if not early_stop_enc_dec:
                        log_msg += f" StdEncDec L: {loss_std.item():.4f} |"
                    if not early_stop_jepa:
                        weighted_aux_loss_str = f"{(current_loss_jepa_aux.item() * aux_loss_weight):.4f}" if aux_loss_fn else "N/A"
                        log_msg += (f" JEPA Pred L: {loss_jepa_pred.item():.4f},"
                                    f" {aux_loss_name} AuxRawL: {current_loss_jepa_aux.item():.4f} (W: {weighted_aux_loss_str}),"
                                    f" Total JEPA L: {total_loss_jepa.item():.4f}")
                    print(log_msg)

        # === Validation Phase ===
        if validation_split > 0 and len(val_dataloader) > 0:
            std_enc_dec.eval()
            jepa_model.eval()
            if hasattr(aux_loss_fn, 'eval'):  # Set DINO loss to eval mode
                aux_loss_fn.eval()

            epoch_val_loss_std = 0
            epoch_val_loss_jepa_pred = 0
            epoch_val_loss_jepa_aux = 0

            with torch.no_grad():
                for s_t_val, a_t_val, r_t_val, s_t_plus_1_val in val_dataloader:
                    s_t_val, s_t_plus_1_val = s_t_val.to(
                        device), s_t_plus_1_val.to(device)
                    if action_type == 'discrete':
                        if a_t_val.ndim == 1:
                            a_t_val = a_t_val.unsqueeze(1)
                        a_t_val_processed = F.one_hot(
                            a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                    else:
                        a_t_val_processed = a_t_val.float().to(device)

                    # Encoder/Decoder Validation
                    if not early_stop_enc_dec:
                        predicted_s_t_plus_1_val = std_enc_dec(
                            s_t_val, a_t_val_processed)
                        val_loss_std = mse_loss_fn(
                            predicted_s_t_plus_1_val, s_t_plus_1_val)
                        epoch_val_loss_std += val_loss_std.item()

                    # JEPA Validation
                    if not early_stop_jepa:
                        pred_emb_val, target_emb_detached_val, online_s_t_emb_val, online_s_t_plus_1_emb_val = jepa_model(
                            s_t_val, a_t_val_processed, s_t_plus_1_val)
                        val_loss_jepa_pred = mse_loss_fn(
                            pred_emb_val, target_emb_detached_val)

                        current_val_loss_jepa_aux = torch.tensor(
                            0.0, device=device)
                        if aux_loss_fn is not None and aux_loss_weight > 0:
                            aux_term_s_t_val, _, _ = aux_loss_fn.calculate_reg_terms(
                                online_s_t_emb_val)
                            aux_term_s_t_plus_1_val, _, _ = aux_loss_fn.calculate_reg_terms(
                                online_s_t_plus_1_emb_val)
                            current_val_loss_jepa_aux = (
                                aux_term_s_t_val + aux_term_s_t_plus_1_val) * 0.5
                        epoch_val_loss_jepa_pred += val_loss_jepa_pred.item()
                        epoch_val_loss_jepa_aux += current_val_loss_jepa_aux.item()

            num_val_batches = len(val_dataloader)
            avg_val_loss_std = epoch_val_loss_std / \
                num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_loss_jepa_pred = epoch_val_loss_jepa_pred / \
                num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_loss_jepa_aux_raw = epoch_val_loss_jepa_aux / \
                num_val_batches if num_val_batches > 0 else float('inf')
            avg_total_val_loss_jepa = avg_val_loss_jepa_pred + \
                avg_val_loss_jepa_aux_raw * aux_loss_weight

            print(f"--- Epoch {epoch+1} Validation Summary ---")
            if not early_stop_enc_dec:
                print(f"  Avg Val StdEncDec L: {avg_val_loss_std:.4f}")
            if not early_stop_jepa:
                print(
                    f"  Avg Val JEPA Pred L: {avg_val_loss_jepa_pred:.4f}, {aux_loss_name} AuxRawL: {avg_val_loss_jepa_aux_raw:.4f}, Total Val JEPA L: {avg_total_val_loss_jepa:.4f}")

            # Early Stopping Checks
            if not early_stop_enc_dec:
                if avg_val_loss_std < best_val_loss_enc_dec - delta:
                    best_val_loss_enc_dec = avg_val_loss_std
                    torch.save(std_enc_dec.state_dict(),
                               checkpoint_path_enc_dec)
                    epochs_no_improve_enc_dec = 0
                    print(
                        f"  Encoder/Decoder: Val loss improved. Saved model to {checkpoint_path_enc_dec}")
                else:
                    epochs_no_improve_enc_dec += 1
                    print(
                        f"  Encoder/Decoder: No val improvement for {epochs_no_improve_enc_dec} epochs.")
                    if epochs_no_improve_enc_dec >= patience:
                        early_stop_enc_dec = True
                        print("  Encoder/Decoder: Early stopping triggered.")

            if not early_stop_jepa:
                if avg_total_val_loss_jepa < best_val_loss_jepa - delta:
                    best_val_loss_jepa = avg_total_val_loss_jepa
                    torch.save(jepa_model.state_dict(), checkpoint_path_jepa)
                    epochs_no_improve_jepa = 0
                    print(
                        f"  JEPA: Val loss improved. Saved model to {checkpoint_path_jepa}")
                else:
                    epochs_no_improve_jepa += 1
                    print(
                        f"  JEPA: No val improvement for {epochs_no_improve_jepa} epochs.")
                    if epochs_no_improve_jepa >= patience:
                        early_stop_jepa = True
                        print("  JEPA: Early stopping triggered.")

        # --- Reward MLP Training Loop ---
        # This section needs to use train_dataloader and be conditional on models not being stopped
        num_batches_reward_train = 0  # Reset for each epoch
        if (reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and not early_stop_enc_dec) or \
           (reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and not early_stop_jepa):
            print(f"Epoch {epoch+1} - Starting Reward MLP Training...")

            std_enc_dec.eval()  # Models providing inputs should be in eval mode
            jepa_model.eval()
            if reward_mlp_enc_dec:
                reward_mlp_enc_dec.train()  # The MLP being trained is in train mode
            if reward_mlp_jepa:
                reward_mlp_jepa.train()

            # Use train_dataloader
            for reward_batch_idx, (s_t_reward, a_t_reward, r_t_reward, s_t_plus_1_reward) in enumerate(train_dataloader):
                s_t_reward = s_t_reward.to(device)
                # a_t_reward_original = a_t_reward # Keep original for potential non-one-hot use if needed
                r_t_reward = r_t_reward.to(device).float().unsqueeze(1)
                # Not strictly needed for all reward MLPs but good for consistency
                s_t_plus_1_reward = s_t_plus_1_reward.to(device)

                # Process actions for reward MLP input generation
                if action_type == 'discrete':
                    if a_t_reward.ndim == 1:
                        a_t_reward = a_t_reward.unsqueeze(1)
                    a_t_reward_processed = F.one_hot(
                        a_t_reward.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_reward_processed = a_t_reward.float().to(device)

                num_batches_reward_train += 1  # Increment for averaging reward losses correctly

                # Encoder/Decoder Reward MLP Training
                if reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and not early_stop_enc_dec:
                    optimizer_reward_mlp_enc_dec.zero_grad()
                    with torch.no_grad():
                        predicted_s_t_plus_1_for_reward = std_enc_dec(
                            s_t_reward, a_t_reward_processed).detach()

                    if enc_dec_mlp_config.get('input_type') == "flatten":
                        input_enc_dec_reward_mlp = predicted_s_t_plus_1_for_reward.view(
                            predicted_s_t_plus_1_for_reward.size(0), -1)
                    else:
                        input_enc_dec_reward_mlp = predicted_s_t_plus_1_for_reward.view(
                            predicted_s_t_plus_1_for_reward.size(0), -1)

                    pred_reward_enc_dec = reward_mlp_enc_dec(
                        input_enc_dec_reward_mlp)
                    loss_reward_enc_dec = mse_loss_fn(
                        pred_reward_enc_dec, r_t_reward)
                    loss_reward_enc_dec.backward()
                    optimizer_reward_mlp_enc_dec.step()
                    epoch_loss_reward_enc_dec += loss_reward_enc_dec.item()

                    if (reward_batch_idx + 1) % enc_dec_mlp_config.get('log_interval', 50) == 0:
                        print(
                            f"  Epoch {epoch+1}, Reward MLP (Enc-Dec) Batch {reward_batch_idx+1}/{num_train_batches}: Loss {loss_reward_enc_dec.item():.4f}")

                # JEPA Reward MLP Training
                if reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and not early_stop_jepa:
                    optimizer_reward_mlp_jepa.zero_grad()
                    with torch.no_grad():
                        # Input to JEPA reward MLP is typically the latent embedding of s_t or s_t+1
                        # Using online encoder's output for s_t_reward as input
                        # Assuming jepa_model.encoder is accessible and is the online encoder
                        # input_jepa_reward_mlp = jepa_model.encoder(s_t_reward, a_t_reward_processed).detach()
                        # OR, if using the JEPA's predicted embedding of the next state:
                        pred_emb_for_reward, _, _, _ = jepa_model(
                            s_t_reward, a_t_reward_processed, s_t_plus_1_reward)  # s_t_plus_1_reward is used by JEPA
                        input_jepa_reward_mlp = pred_emb_for_reward.detach()

                    pred_reward_jepa = reward_mlp_jepa(input_jepa_reward_mlp)
                    loss_reward_jepa = mse_loss_fn(
                        pred_reward_jepa, r_t_reward)
                    loss_reward_jepa.backward()
                    optimizer_reward_mlp_jepa.step()
                    epoch_loss_reward_jepa += loss_reward_jepa.item()

        # After all training for the epoch (main models + reward MLPs)
        avg_loss_std = epoch_loss_std / \
            num_train_batches if not early_stop_enc_dec and num_train_batches > 0 else 0
        avg_loss_jepa_pred = epoch_loss_jepa_pred / \
            num_train_batches if not early_stop_jepa and num_train_batches > 0 else 0
        avg_loss_jepa_aux_raw = epoch_loss_jepa_aux / \
            num_train_batches if not early_stop_jepa and num_train_batches > 0 else 0

        avg_loss_reward_enc_dec = epoch_loss_reward_enc_dec / \
            num_batches_reward_train if num_batches_reward_train > 0 and not early_stop_enc_dec else 0
        avg_loss_reward_jepa = epoch_loss_reward_jepa / \
            num_batches_reward_train if num_batches_reward_train > 0 and not early_stop_jepa else 0

        print(
            f"--- Epoch {epoch+1}/{config.get('num_epochs', 10)} Training Summary ---")
        if not early_stop_enc_dec:
            print(f"  Avg Train StdEncDec L: {avg_loss_std:.4f}")
        if not early_stop_jepa:
            avg_total_jepa_loss = avg_loss_jepa_pred + \
                avg_loss_jepa_aux_raw * aux_loss_weight
            print(
                f"  Avg Train JEPA Pred L: {avg_loss_jepa_pred:.4f}, {aux_loss_name} AuxRawL: {avg_loss_jepa_aux_raw:.4f}, Total Train JEPA L: {avg_total_jepa_loss:.4f}")
        if enc_dec_mlp_config.get('enabled', False) and reward_mlp_enc_dec and not early_stop_enc_dec:
            print(
                f"  Avg Train Reward MLP (Enc-Dec) L: {avg_loss_reward_enc_dec:.4f}")
        if jepa_mlp_config.get('enabled', False) and reward_mlp_jepa and not early_stop_jepa:
            print(
                f"  Avg Train Reward MLP (JEPA) L: {avg_loss_reward_jepa:.4f}")

        if early_stop_enc_dec and early_stop_jepa:
            print("Both models triggered early stopping. Ending training.")
            break

    print("Main training loop finished.")

    # Load best models
    print("Loading best models...")
    if os.path.exists(checkpoint_path_enc_dec):
        print(
            f"Loading best Encoder/Decoder model from {checkpoint_path_enc_dec}")
        std_enc_dec.load_state_dict(torch.load(
            checkpoint_path_enc_dec, map_location=device))
    else:
        print(
            f"No checkpoint found for Encoder/Decoder at {checkpoint_path_enc_dec}. Using last trained model.")
    std_enc_dec.eval()

    if os.path.exists(checkpoint_path_jepa):
        print(f"Loading best JEPA model from {checkpoint_path_jepa}")
        jepa_model.load_state_dict(torch.load(
            checkpoint_path_jepa, map_location=device))
    else:
        print(
            f"No checkpoint found for JEPA at {checkpoint_path_jepa}. Using last trained model.")
    jepa_model.eval()

    # Potentially set aux_loss_fn to eval if it has state and is used post-training
    if hasattr(aux_loss_fn, 'eval'):
        aux_loss_fn.eval()

    # Also set reward MLPs to eval mode
    if reward_mlp_enc_dec:
        reward_mlp_enc_dec.eval()
    if reward_mlp_jepa:
        reward_mlp_jepa.eval()

    print("Process complete. Models are in eval mode with best weights loaded (if available).")


if __name__ == '__main__':
    main()
