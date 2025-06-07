import torch
import os # For path joining and checking existence

# Import functions from the new modules in the 'src' directory
from src.config_utils import load_config
from src.env_utils import get_env_details
from src.data_handling import prepare_dataloaders
from src.model_setup import initialize_models
from src.loss_setup import initialize_loss_functions
from src.optimizer_setup import initialize_optimizers
from src.training_engine import run_training_epochs

def main():
    # 1. Load Configuration
    # config.yaml is expected to be in the same directory as this main.py (project root)
    config = load_config(config_path='config.yaml')

    # 2. Setup Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # 3. Get Environment Details
    action_dim, action_type, observation_space = get_env_details(config['environment_name'])

    # 4. Prepare Dataloaders
    # Validation split ratio from early stopping config or general data config
    early_stopping_config = config.get('early_stopping', {})
    validation_split = early_stopping_config.get('validation_split', 0.2)
    dataloaders_map = {}
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    if train_dataloader is None:
        print("Exiting due to no training data.")
        return
    dataloaders_map['train'] = train_dataloader
    if val_dataloader:
        dataloaders_map['val'] = val_dataloader

    # 5. Initialize Models
    image_h_w = config['image_size']
    input_channels = config.get('input_channels', 3)
    models_map = initialize_models(config, action_dim, device, image_h_w, input_channels)

    # Extract models for convenience if needed, or just pass models_map
    std_enc_dec = models_map.get('std_enc_dec')
    jepa_model = models_map.get('jepa')
    # reward_mlp_enc_dec = models_map.get('reward_mlp_enc_dec') # available in map
    # reward_mlp_jepa = models_map.get('reward_mlp_jepa')       # available in map

    # 6. Initialize Loss Functions
    # Pass jepa_model.latent_dim if DINO loss is configured
    jepa_model_latent_dim_for_dino = None
    if jepa_model and config.get('auxiliary_loss', {}).get('type') == 'dino':
        jepa_model_latent_dim_for_dino = jepa_model.latent_dim # Accessing from initialized model

    losses_map = initialize_loss_functions(config, device, jepa_model_latent_dim=jepa_model_latent_dim_for_dino)

    # 7. Initialize Optimizers
    optimizers_map = initialize_optimizers(models_map, config)

    # 8. Run Training Epochs
    # The training engine will handle early stopping and saving best models internally
    run_training_epochs(
        models_map=models_map,
        optimizers_map=optimizers_map,
        losses_map=losses_map,
        dataloaders_map=dataloaders_map,
        device=device,
        config=config, # Pass the full config for various parameters like num_epochs, log_interval etc.
        action_dim=action_dim,
        action_type=action_type,
        image_h_w=image_h_w, # Needed for reward MLP input calculation within training_engine
        input_channels=input_channels # Needed for reward MLP input calculation
    )

    # 9. Post-Training: Load best models and set to eval mode
    # Checkpoint paths from config (used by training_engine for saving, here for loading)
    checkpoint_path_enc_dec = early_stopping_config.get('checkpoint_path_enc_dec', 'best_encoder_decoder.pth')
    checkpoint_path_jepa = early_stopping_config.get('checkpoint_path_jepa', 'best_jepa.pth')

    print("Loading best models (if available) after training...")
    if std_enc_dec and os.path.exists(checkpoint_path_enc_dec):
        print(f"Loading best Encoder/Decoder model from {checkpoint_path_enc_dec}")
        std_enc_dec.load_state_dict(torch.load(checkpoint_path_enc_dec, map_location=device))
    elif std_enc_dec:
        print(f"No checkpoint found for Encoder/Decoder at {checkpoint_path_enc_dec}. Model remains in its last training state.")

    # Ensure model is in eval mode even if no checkpoint was loaded
    if std_enc_dec:
        std_enc_dec.eval()

    if jepa_model and os.path.exists(checkpoint_path_jepa):
        print(f"Loading best JEPA model from {checkpoint_path_jepa}")
        jepa_model.load_state_dict(torch.load(checkpoint_path_jepa, map_location=device))
    elif jepa_model:
        print(f"No checkpoint found for JEPA at {checkpoint_path_jepa}. Model remains in its last training state.")

    if jepa_model:
        jepa_model.eval()

    # Also set reward MLPs to eval mode if they exist
    if models_map.get('reward_mlp_enc_dec'):
        models_map['reward_mlp_enc_dec'].eval()
    if models_map.get('reward_mlp_jepa'):
        models_map['reward_mlp_jepa'].eval()

    # Set auxiliary loss function to eval mode if applicable (e.g., DINO's center update)
    if losses_map.get('aux_fn') and hasattr(losses_map['aux_fn'], 'eval'):
        losses_map['aux_fn'].eval()

    print("Process complete. Models are in eval mode (with best weights loaded if checkpoints were found).")

if __name__ == '__main__':
    main()
