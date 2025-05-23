import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np # For action space handling
import torch.nn.functional as F # Moved to top

# Our project modules
from utils.data_utils import collect_random_episodes, ExperienceDataset
from models.vit import ViT # Though ViT is used by models, not directly here
from models.encoder_decoder import StandardEncoderDecoder
from models.jepa import JEPA
from utils.losses import VICRegLoss

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_env_details(env_name):
    # Create a temporary env to get action and observation space details
    # This is crucial for setting up model input/output dimensions correctly.
    temp_env = gym.make(env_name)
    action_space = temp_env.action_space
    observation_space = temp_env.observation_space # This is the raw observation space
    
    # Determine action_dim
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        action_type = 'discrete'
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
        action_type = 'continuous'
    else:
        temp_env.close() # Close env before raising error
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

    # For image_size, we rely on config, but input_channels might be inferred
    # if observation is image. Defaulting to 3 (RGB) for now if not Box.
    # The ViT expects C, H, W. Our data collection resizes to (image_size, image_size)
    # and ToTensor converts to C, H, W.
    # We assume 3 channels for typical RGB gym environments.
    # If it's grayscale, config should specify input_channels=1.
    # For now, let's assume config handles input_channels.
    
    temp_env.close()
    print(f"Environment: {env_name}")
    print(f"Action space type: {action_type}, Action dimension: {action_dim}")
    print(f"Raw observation space: {observation_space}")
    return action_dim, action_type, observation_space


def main():
    config = load_config()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get environment details
    action_dim, action_type, obs_space = get_env_details(config['environment_name'])
    
    # For environments like 'CarRacing-v2', observations are Box(96, 96, 3)
    # Our ViT expects a fixed image_size from config, and data_utils resizes to it.
    # input_channels should ideally come from config or be inferred if possible.
    # Default to 3 for now, assuming RGB.
    input_channels = config.get('input_channels', 3) 
    image_h_w = config['image_size'] # This is a single int for square images

    # Data Collection
    print("Starting data collection...")
    dataset = collect_random_episodes(
        env_name=config['environment_name'],
        num_episodes=config['num_episodes_data_collection'],
        max_steps_per_episode=config['max_steps_per_episode_data_collection'],
        image_size=(image_h_w, image_h_w) # Pass as tuple
    )
    
    if len(dataset) == 0:
        print("No data collected. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    # --- Standard Encoder-Decoder Model ---
    print("Initializing Standard Encoder-Decoder Model...")
    std_enc_dec = StandardEncoderDecoder(
        image_size=image_h_w,
        patch_size=config['patch_size'],
        input_channels=input_channels,
        action_dim=action_dim, # Correctly passed
        action_emb_dim=config.get('action_emb_dim', config['latent_dim']), # Default if not set
        latent_dim=config['latent_dim'],
        decoder_dim=config['decoder_dim'],
        decoder_depth=config['num_decoder_layers'],
        decoder_heads=config['num_heads'],
        decoder_mlp_dim=config['mlp_dim'],
        output_channels=input_channels, # Predict same number of channels
        output_image_size=image_h_w,
        vit_depth=config['num_encoder_layers'],
        vit_heads=config['num_heads'],
        vit_mlp_dim=config['mlp_dim']
    ).to(device)
    optimizer_std_enc_dec = optim.AdamW(std_enc_dec.parameters(), lr=config['learning_rate'])
    mse_loss_fn = nn.MSELoss()

    # --- JEPA Model ---
    print("Initializing JEPA Model...")
    jepa_model = JEPA(
        image_size=image_h_w,
        patch_size=config['patch_size'],
        input_channels=input_channels,
        action_dim=action_dim, # Correctly passed
        action_emb_dim=config.get('action_emb_dim', config['latent_dim']),
        latent_dim=config['latent_dim'],
        predictor_hidden_dim=config['jepa_predictor_hidden_dim'],
        predictor_output_dim=config['latent_dim'], # Must match latent_dim
        vit_depth=config['num_encoder_layers'],
        vit_heads=config['num_heads'],
        vit_mlp_dim=config['mlp_dim'],
        ema_decay=config['ema_decay']
    ).to(device)
    optimizer_jepa = optim.AdamW(jepa_model.parameters(), lr=config['learning_rate']) # Separate optimizer
    vicreg_loss_fn = VICRegLoss(
        sim_coeff=config['vicreg_sim_coeff'], # Not used directly if using calculate_reg_terms
        std_coeff=config['vicreg_std_coeff'],
        cov_coeff=config['vicreg_cov_coeff']
    ).to(device)

    # --- Training Loop ---
    print(f"Starting training for {config['num_epochs']} epochs...")
    for epoch in range(config['num_epochs']):
        epoch_loss_std = 0
        epoch_loss_jepa_pred = 0
        epoch_loss_jepa_vicreg = 0
        
        for batch_idx, (s_t, a_t, s_t_plus_1) in enumerate(dataloader):
            s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
            # Action tensor a_t needs to be compatible with action_embedding layer in models.
            # Our data_utils creates action tensors as float.
            # If action_type is 'discrete', and model expects one-hot, it needs conversion.
            # StandardEncoderDecoder & JEPA use nn.Linear for action_embedding, so float is fine.
            # If action is discrete (e.g. int from env.sample()), it needs one-hot encoding
            # if action_dim in model is the number of discrete actions.
            # Let's assume action_dim from get_env_details is already correct for nn.Linear.
            # If action_type is 'discrete', a_t from dataloader is (batch_size,). Needs to be (batch_size, 1) for some linear layers
            # or one-hot encoded to (batch_size, action_dim).
            # The current data_utils.py creates action_tensor = torch.tensor(action, dtype=torch.float32)
            # If action is an int, this is (batch_size,). If it's a numpy array (e.g. continuous), it's (batch_size, num_continuous_actions)

            if action_type == 'discrete':
                # If action_dim is num_classes for discrete, nn.Linear needs one-hot input.
                # Or, the models' action_embedding should be nn.Embedding for discrete indices.
                # Given we used nn.Linear(action_dim, ...), we should one-hot discrete actions.
                # a_t is currently (batch_size,). Needs to be (batch_size, action_dim)
                # a_t_one_hot = F.one_hot(a_t.long().squeeze(-1), num_classes=action_dim).float().to(device)
                # Squeeze -1 if a_t is (batch_size, 1) from dataloader for discrete actions.
                # If a_t from dataloader is already (batch_size,), then a_t.long() is fine.
                # Let's check shape of a_t from dataloader for discrete actions.
                # It's likely (batch_size,) if env.action_space.sample() returns a scalar int.
                # If so, a_t.long() is correct.
                if a_t.ndim == 1 or (a_t.ndim == 2 and a_t.shape[1] == 1):
                    a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else: # Should not happen for discrete if data is collected correctly
                    a_t_processed = a_t.float().to(device) # Fallback
            else: # Continuous
                a_t_processed = a_t.float().to(device)


            # --- Standard Encoder-Decoder Training Step ---
            optimizer_std_enc_dec.zero_grad()
            predicted_s_t_plus_1 = std_enc_dec(s_t, a_t_processed)
            loss_std = mse_loss_fn(predicted_s_t_plus_1, s_t_plus_1)
            loss_std.backward()
            optimizer_std_enc_dec.step()
            epoch_loss_std += loss_std.item()

            # --- JEPA Training Step ---
            optimizer_jepa.zero_grad()
            # Forward pass: pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb
            pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = jepa_model(s_t, a_t_processed, s_t_plus_1)
            
            # Prediction Loss (MSE between predicted embedding and target online embedding of s_t+1)
            loss_jepa_pred = mse_loss_fn(pred_emb, target_emb_detached)
            
            # VICReg Loss on online encoder outputs
            # Apply to concatenated embeddings from s_t and s_t+1, or separately and sum/average
            # Let's apply to both online_s_t_emb and online_s_t_plus_1_emb and average the reg loss.
            # The VICRegLoss class's calculate_reg_terms takes one batch of embeddings.
            reg_loss_s_t, _, _ = vicreg_loss_fn.calculate_reg_terms(online_s_t_emb)
            reg_loss_s_t_plus_1, _, _ = vicreg_loss_fn.calculate_reg_terms(online_s_t_plus_1_emb)
            loss_jepa_vicreg = (reg_loss_s_t + reg_loss_s_t_plus_1) * 0.5
            
            total_loss_jepa = loss_jepa_pred + loss_jepa_vicreg
            total_loss_jepa.backward()
            optimizer_jepa.step()
            
            # Update JEPA Target Encoder (EMA)
            jepa_model.update_target_network()
            
            epoch_loss_jepa_pred += loss_jepa_pred.item()
            epoch_loss_jepa_vicreg += loss_jepa_vicreg.item()

            if batch_idx % 50 == 0: # Log every 50 batches
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"StdEncDec Loss: {loss_std.item():.4f} | "
                      f"JEPA Pred Loss: {loss_jepa_pred.item():.4f}, VICReg Loss: {loss_jepa_vicreg.item():.4f}")

        avg_loss_std = epoch_loss_std / len(dataloader)
        avg_loss_jepa_pred = epoch_loss_jepa_pred / len(dataloader)
        avg_loss_jepa_vicreg = epoch_loss_jepa_vicreg / len(dataloader)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} Summary:")
        print(f"  Avg Standard Encoder-Decoder Loss: {avg_loss_std:.4f}")
        print(f"  Avg JEPA Prediction Loss: {avg_loss_jepa_pred:.4f}")
        print(f"  Avg JEPA VICReg Loss: {avg_loss_jepa_vicreg:.4f}")
        print(f"  Avg Total JEPA Loss: {(avg_loss_jepa_pred + avg_loss_jepa_vicreg):.4f}")

    print("Training finished.")

    # TODO: Add evaluation logic here if needed
    # For example, save models, generate sample predictions, etc.

if __name__ == '__main__':
    # For one-hot encoding if needed in training loop - F is now imported at top
    main()

```
