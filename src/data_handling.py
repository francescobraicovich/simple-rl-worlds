# Contents for src/data_handling.py
import torch
from torch.utils.data import DataLoader
from src.utils.data_utils import collect_random_episodes, collect_ppo_episodes # Absolute import from src
from src.utils.config_utils import get_effective_input_channels, validate_environment_config

def prepare_dataloaders(config, validation_split): # validation_split is from config['data']['validation_split'] passed from main.py
    print("Starting data collection...")

    # Validate environment configuration and show effective parameters
    validate_environment_config(config)

    # Accessing nested configuration parameters
    env_config = config.get('environment', {})
    training_config = config.get('training', {})
    data_collection_config = config.get('data', {}).get('collection', {})
    ppo_agent_config = config.get('ppo_agent', {})

    image_h = env_config.get('image_height')
    image_w = env_config.get('image_width')
    if image_h is None or image_w is None:
        raise ValueError("Configuration error: 'environment.image_height' or 'environment.image_width' is not set.")
    img_size_tuple = (image_h, image_w)

    frame_skipping = training_config.get('frame_skipping', 0)
    
    max_steps = data_collection_config.get('max_steps_per_episode', 200) # Default from old code was 200 if key_max_steps not found

    use_ppo = ppo_agent_config.get('enabled', False)

    # The functions collect_ppo_episodes and collect_random_episodes are passed the full 'config' object.
    # They will need to be refactored internally to use the new config structure.
    # This current refactoring of prepare_dataloaders only addresses direct accesses within this function.
    if use_ppo:
        print("Using PPO agent for data collection.")
        train_dataset, val_dataset = collect_ppo_episodes(
            config=config, # Pass the full config object -  data_utils.py needs update
            max_steps_per_episode=max_steps,
            image_size=img_size_tuple,
            validation_split_ratio=validation_split, # This comes from config['data']['validation_split']
            frame_skipping=frame_skipping
        )
    else:
        print("Using random actions for data collection.")
        train_dataset, val_dataset = collect_random_episodes(
            config=config, # Pass the full config object - data_utils.py needs update
            max_steps_per_episode=max_steps,
            image_size=img_size_tuple,
            validation_split_ratio=validation_split, # This comes from config['data']['validation_split']
            frame_skipping=frame_skipping
        )

    if len(train_dataset) == 0:
        print("No training data collected. Exiting program or handling as error.")
        return None, None

    if validation_split > 0 and len(val_dataset) == 0:
        print("Warning: Validation split is > 0 but no validation data was collected. Check dataset size and split ratio.")

    batch_size = training_config.get('batch_size', 32)
    num_workers = training_config.get('num_workers', 4)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_dataloader = None
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_dataloader, val_dataloader
