import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gymnasium as gym
import torch.nn.functional as F

from utils.data_utils import collect_random_episodes, ExperienceDataset
from models.encoder_decoder import StandardEncoderDecoder
from models.jepa import JEPA
from utils.losses import VICRegLoss

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
    print(f"Raw observation space: {observation_space}")
    return action_dim, action_type, observation_space

def main():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    action_dim, action_type, _ = get_env_details(config['environment_name'])
    
    input_channels = config.get('input_channels', 3)
    image_h_w = config['image_size']

    # Encoder configuration
    encoder_type = config.get('encoder_type', 'vit')
    all_encoder_params_from_config = config.get('encoder_params', {})
    # Select the sub-dictionary for the chosen encoder_type, ensuring it's a dict
    specific_encoder_params = all_encoder_params_from_config.get(encoder_type, {})
    if specific_encoder_params is None: # Handles case where key exists but value is null
        specific_encoder_params = {}


    global_patch_size = config.get('patch_size', 16) # Used by ViT and default for decoder

    # If ViT is the encoder_type, ensure 'patch_size' is set in its specific_encoder_params.
    # The ViT model constructor expects 'patch_size' directly.
    if encoder_type == 'vit':
        if 'patch_size' not in specific_encoder_params or specific_encoder_params['patch_size'] is None:
            specific_encoder_params['patch_size'] = global_patch_size

        # Set other ViT parameters with defaults if not provided in config's vit section.
        # These are parameters the ViT model's __init__ expects.
        # The models themselves also have defaults, but setting here allows config override of those.
        specific_encoder_params.setdefault('depth', config.get('encoder_params', {}).get('vit', {}).get('depth', 6))
        specific_encoder_params.setdefault('heads', config.get('encoder_params', {}).get('vit', {}).get('heads', 8))
        specific_encoder_params.setdefault('mlp_dim', config.get('encoder_params', {}).get('vit', {}).get('mlp_dim', 1024))
        specific_encoder_params.setdefault('pool', config.get('encoder_params', {}).get('vit', {}).get('pool', 'cls'))
        specific_encoder_params.setdefault('dropout', config.get('encoder_params', {}).get('vit', {}).get('dropout', 0.0))
        specific_encoder_params.setdefault('emb_dropout', config.get('encoder_params', {}).get('vit', {}).get('emb_dropout', 0.0))


    print("Starting data collection...")
    dataset = collect_random_episodes(
        env_name=config['environment_name'],
        num_episodes=config.get('num_episodes_data_collection', 50),
        max_steps_per_episode=config.get('max_steps_per_episode_data_collection', 200),
        image_size=(image_h_w, image_h_w)
    )
    
    if len(dataset) == 0:
        print("No data collected. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=config.get('num_workers', 4), pin_memory=True)

    print(f"Initializing Standard Encoder-Decoder Model with {encoder_type.upper()} encoder...")
    std_enc_dec = StandardEncoderDecoder(
        image_size=image_h_w,
        patch_size=global_patch_size, # Passed for ViT encoder & default for decoder. ViT gets from specific_encoder_params if 'vit'
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=config.get('action_emb_dim', config.get('latent_dim', 128)),
        latent_dim=config.get('latent_dim', 128),
        decoder_dim=config.get('decoder_dim', 128),
        decoder_depth=config.get('decoder_depth', config.get('num_decoder_layers', 3)),
        decoder_heads=config.get('decoder_heads', config.get('num_heads', 6)),
        decoder_mlp_dim=config.get('decoder_mlp_dim', config.get('mlp_dim', 256)),
        output_channels=input_channels,
        output_image_size=image_h_w,
        decoder_dropout=config.get('decoder_dropout', 0.0),
        encoder_type=encoder_type,
        encoder_params=specific_encoder_params, # Pass the type-specific params
        decoder_patch_size=config.get('decoder_patch_size', global_patch_size)
    ).to(device)
    optimizer_std_enc_dec = optim.AdamW(std_enc_dec.parameters(), lr=config.get('learning_rate', 0.0003))

    mse_loss_fn = nn.MSELoss()

    print(f"Initializing JEPA Model with {encoder_type.upper()} encoder...")
    jepa_model = JEPA(
        image_size=image_h_w,
        patch_size=global_patch_size, # ViT encoder in JEPA gets patch_size from specific_encoder_params
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=config.get('action_emb_dim', config.get('latent_dim', 128)),
        latent_dim=config.get('latent_dim', 128),
        predictor_hidden_dim=config.get('jepa_predictor_hidden_dim', 256),
        predictor_output_dim=config.get('latent_dim', 128), # Must match latent_dim
        ema_decay=config.get('ema_decay', 0.996),
        encoder_type=encoder_type,
        encoder_params=specific_encoder_params # Pass the type-specific params
    ).to(device)
    optimizer_jepa = optim.AdamW(jepa_model.parameters(), lr=config.get('learning_rate_jepa', config.get('learning_rate', 0.0003)))

    vicreg_loss_fn = VICRegLoss(
        sim_coeff=config.get('vicreg_sim_coeff', 0.0), # Default to 0 as per some VICReg variants
        std_coeff=config.get('vicreg_std_coeff', 25.0),
        cov_coeff=config.get('vicreg_cov_coeff', 1.0)
    ).to(device)

    print(f"Starting training for {config.get('num_epochs', 10)} epochs...")
    for epoch in range(config.get('num_epochs', 10)):
        epoch_loss_std = 0
        epoch_loss_jepa_pred = 0
        epoch_loss_jepa_vicreg = 0
        
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
            
            reg_loss_s_t, _, _ = vicreg_loss_fn.calculate_reg_terms(online_s_t_emb)
            reg_loss_s_t_plus_1, _, _ = vicreg_loss_fn.calculate_reg_terms(online_s_t_plus_1_emb)
            current_loss_jepa_vicreg = (reg_loss_s_t + reg_loss_s_t_plus_1) * 0.5
            
            vicreg_weight = config.get('vicreg_loss_weight', 1.0)
            total_loss_jepa = loss_jepa_pred + current_loss_jepa_vicreg * vicreg_weight
            total_loss_jepa.backward()
            optimizer_jepa.step()
            
            jepa_model.update_target_network()
            
            epoch_loss_jepa_pred += loss_jepa_pred.item()
            epoch_loss_jepa_vicreg += current_loss_jepa_vicreg.item()

            if (batch_idx + 1) % config.get('log_interval', 50) == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}: "
                      f"StdEncDec Loss: {loss_std.item():.4f} | "
                      f"JEPA Pred Loss: {loss_jepa_pred.item():.4f}, VICReg Raw Loss: {current_loss_jepa_vicreg.item():.4f} "
                      f"(Weighted: {(current_loss_jepa_vicreg * vicreg_weight):.4f}), "
                      f"Total JEPA Loss: {total_loss_jepa.item():.4f}")

        avg_loss_std = epoch_loss_std / num_batches
        avg_loss_jepa_pred = epoch_loss_jepa_pred / num_batches
        avg_loss_jepa_vicreg_raw = epoch_loss_jepa_vicreg / num_batches
        
        print(f"Epoch {epoch+1}/{config.get('num_epochs', 10)} Summary:")
        print(f"  Avg Standard Encoder-Decoder Loss: {avg_loss_std:.4f}")
        print(f"  Avg JEPA Prediction Loss: {avg_loss_jepa_pred:.4f}")
        print(f"  Avg JEPA VICReg Loss (Raw): {avg_loss_jepa_vicreg_raw:.4f}")
        avg_total_jepa_loss = avg_loss_jepa_pred + avg_loss_jepa_vicreg_raw * vicreg_weight
        print(f"  Avg Total JEPA Loss: {avg_total_jepa_loss:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    main()
