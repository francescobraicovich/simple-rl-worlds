import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random  # Added for shuffling episodes
import os
import multiprocessing
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from src.rl_agent import create_ppo_agent, train_ppo_agent
from src.utils.env_wrappers import ActionRepeatWrapper, ImagePreprocessingWrapper, FrameStackWrapper
from src.utils.config_utils import create_image_preprocessing_transforms, FrameStackBuffer, get_single_frame_channels
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Distribution checks will be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plot generation disabled.")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Advanced distribution visualization disabled.")

import warnings
# import cv2 # For image resizing - Removed as torchvision.transforms is used


class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, transform=None, config=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.transform = transform
        self.config = config
        if self.config is None:
            # This is a fallback if config is not provided, though the plan is to always provide it.
            # Consider raising an error if config is essential for all uses.
            print("Warning: ExperienceDataset created without a config. State shape validation will be skipped.")


    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.next_states[idx]

        #if self.transform:
        #    state = self.transform(state)
        #    next_state = self.transform(next_state)
#
        #if self.config:
        #    # Get configuration for shape validation
        #    image_height = self.config['environment']['image_height']
        #    image_width = self.config['environment']['image_width']
#
        #    num_channels = self.config['environment'].get('input_channels_per_frame', 3)  # Default to 1 if not set
        #    if self.config['environment'].get('grayscale_conversion', True):
        #        num_channels = 1
#
        #    num_frames = self.config['environment'].get('frame_stack_size')  # Default to 1 if not set
        #    
        #    expected_stacked_shape = (num_frames, num_channels, image_height, image_width)
#
        #    # Validate current state (should be stacked)
        #    if state.shape != expected_stacked_shape:
        #        error_msg = f"State shape {state.shape} does not match expected stacked shape {expected_stacked_shape} for idx {idx}."
        #        raise ValueError(error_msg)
#
        #    # Validate next state (should be single frame)
        #    if next_state.shape != expected_stacked_shape:
        #        error_msg = f"Next state shape {next_state.shape} does not match expected stacked frame shape {expected_stacked_shape} for idx {idx}."
        #        raise ValueError(error_msg)

        # Convert action to tensor, ensure it's float for potential nn.Linear embedding
        # Adjust dtype based on action type (discrete typically long, continuous float)
        # For simplicity, let's assume actions will be made float.
        # If discrete, they might be indices; ensure they are handled appropriately later (e.g. one-hot or embedding layer).
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        return state, action_tensor, reward_tensor, next_state


def collect_random_episodes(config, max_steps_per_episode, image_size, validation_split_ratio, frame_skipping):
    env_name = config['environment']['name']
    num_episodes = config['data']['collection']['num_episodes']

    # New configuration keys
    dataset_dir = config['data']['dataset']['dir']
    load_dataset_filename = config['data']['dataset']['load_path'] # Filename or path relative to dataset_dir
    save_dataset_filename = config['data']['dataset']['filename'] # Filename for saving new datasets

    os.makedirs(dataset_dir, exist_ok=True)

    data_loaded_successfully = False
    if load_dataset_filename: # If not an empty string, attempt to load
        dataset_path = os.path.join(dataset_dir, load_dataset_filename)
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}...")
            try:
                data = torch.load(dataset_path, weights_only=False)

                loaded_train_dataset = data['train_dataset']
                loaded_val_dataset = data['val_dataset']
                metadata = data['metadata']
                print(f"Loaded metadata: {metadata}") # Print loaded metadata
                loaded_frame_skipping = metadata.get('frame_skipping', 'N/A')
                print(f"Frame skipping from loaded metadata: {loaded_frame_skipping}")


                loaded_env_name = metadata.get('environment_name')
                if loaded_env_name != env_name:
                    print(f"Error: Mismatch between loaded dataset environment ('{loaded_env_name}') and config environment ('{env_name}').")
                    raise ValueError("Environment mismatch in loaded dataset.")

                print(f"Successfully loaded dataset for environment '{loaded_env_name}' with {metadata.get('num_episodes_collected', 'N/A')} episodes from {dataset_path}.")
                data_loaded_successfully = True
                return loaded_train_dataset, loaded_val_dataset
            except Exception as e:
                print(f"Error loading dataset from {dataset_path}: {e}. Proceeding to data collection.")
        else:
            print(f"Warning: Dataset {dataset_path} not found. Proceeding to data collection.")
    else:
        print("`data.dataset.load_path` is empty. Proceeding to data collection.")

    # If data was not loaded, proceed with data collection
    print(f"Collecting data from environment: {env_name}")
    # Try to make the environment, prioritizing 'rgb_array' for image collection,
    # as this function is designed to feed an image-based preprocessing pipeline.
    try:
        env = gym.make(env_name, render_mode='rgb_array')
        print(
            f"Successfully created env '{env_name}' with render_mode='rgb_array'.")
    except Exception as e_rgb:
        print(
            f"Failed to create env '{env_name}' with render_mode='rgb_array': {e_rgb}. Trying with render_mode=None...")
        try:
            env = gym.make(env_name, render_mode=None)
            print(
                f"Successfully created env '{env_name}' with render_mode=None.")
        except Exception as e_none:
            print(
                f"Failed to create env '{env_name}' with render_mode=None: {e_none}. Trying without render_mode arg...")
            env = gym.make(env_name)  # Fallback
            print(
                f"Successfully created env '{env_name}' with default render_mode ('{env.render_mode if hasattr(env, 'render_mode') else 'unknown'}').")

    all_episodes_raw_data = []  # Stores list of lists of (s,a,r,s') tuples

    # Create preprocessing transforms using the new config-based approach
    preprocess = create_image_preprocessing_transforms(config, image_size)
    
    # Get frame stacking configuration
    env_config = config.get('environment', {})
    frame_stack_size = env_config.get('frame_stack_size', 1)
    
    # Determine single frame shape after preprocessing
    single_frame_channels = get_single_frame_channels(config)
    single_frame_shape = (single_frame_channels, image_size[0], image_size[1])

    for episode_idx in range(num_episodes):
        current_state_img, info = env.reset()
        # Standardize image check after reset
        # If the observation is not a uint8 numpy array, it might be a float array (like CartPole)
        # or something else. Try to render if the env is in 'rgb_array' mode.
        initial_obs_is_uint8_image = isinstance(
            current_state_img, np.ndarray) and current_state_img.dtype == np.uint8

        if not initial_obs_is_uint8_image:
            # If initial obs is not a uint8 image, and env is in 'rgb_array' mode, try to render.
            if env.render_mode == 'rgb_array':
                print(
                    f"Warning: Initial observation for episode {episode_idx+1} is not uint8. Attempting render due to env.render_mode='rgb_array'. Original obs type: {type(current_state_img)}, dtype: {current_state_img.dtype if hasattr(current_state_img, 'dtype') else 'N/A'}")
                current_state_img = env.render()  # This should now provide an image
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8):
                    print(
                        f"Error: env.render() in rgb_array mode did not return a uint8 numpy array for episode {episode_idx+1}. State: {current_state_img}. Skipping episode.")
                    continue
            else:
                # If not uint8 and not in rgb_array mode, we can't process it as an image.
                print(
                    f"Warning: Initial observation for episode {episode_idx+1} is not a uint8 image and env.render_mode is '{env.render_mode}'. Cannot process as image. Skipping episode. Observation: {current_state_img}")
                continue

        # After potential rendering, re-check if we have a valid image structure (at least 2D)
        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
            print(
                f"Skipping episode {episode_idx+1} due to unsuitable initial state after potential render. State: {current_state_img}")
            continue

        # Initialize frame stacking buffer for this episode
        frame_buffer = FrameStackBuffer(frame_stack_size, single_frame_shape)
        
        # Process the initial frame and add to buffer
        initial_frame_pil = preprocess(current_state_img)
        # Convert PIL to tensor without normalization (keep 0-255 range)
        initial_frame = torch.from_numpy(np.array(initial_frame_pil, dtype=np.uint8)).float()
        # Ensure CHW format
        if len(initial_frame.shape) == 2:
            initial_frame = initial_frame.unsqueeze(0)  # Add channel dimension
        elif len(initial_frame.shape) == 3:
            initial_frame = initial_frame.permute(2, 0, 1)  # HWC to CHW
        frame_buffer.add_frame(initial_frame)
        
        # Get the initial stacked state
        current_stacked_state = frame_buffer.get_stacked_frames()

        episode_transitions = []
        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0  # Initialize cumulative reward

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            # Record the current stacked state before taking action
            recorded_current_stacked_state = current_stacked_state
            accumulated_reward = 0.0

            action = env.action_space.sample() # Primary action
            print('Action sampled:', action)

            # First step
            next_state_img, reward, terminated, truncated, info = env.step(action)
            accumulated_reward += reward
            current_step_next_state_img = next_state_img
            # Primary step counts towards step_count here for the main loop condition,
            # but the actual recorded transition will use the state *after* skipping.
            # The step_count increment for the main recorded transition happens *after* skipping block.

            if frame_skipping > 0:
                for _ in range(frame_skipping):
                    if terminated or truncated:
                        break

                    # Increment step_count for each actual environment step *during* skipping
                    step_count += 1 # This step is part of the episode's max_steps
                    if step_count >= max_steps_per_episode:
                        break

                    action_skip = env.action_space.sample()
                    next_state_skip, reward_skip, terminated, truncated, info = env.step(action_skip)
                    accumulated_reward += reward_skip
                    current_step_next_state_img = next_state_skip

            # Now, current_step_next_state_img holds S' after all skips
            # and accumulated_reward has the sum of rewards over the skipped frames.
            # The action is the original action taken from recorded_current_state_img.
            next_state_img = current_step_next_state_img # Use this for subsequent checks and storage

            # Standardize image check for next_state (which is current_step_next_state_img)
            next_obs_is_uint8_image = isinstance(
                next_state_img, np.ndarray) and next_state_img.dtype == np.uint8

            if not next_obs_is_uint8_image:
                if env.render_mode == 'rgb_array':
                    # print(f"Debug: Next state obs for ep {episode_idx+1}, step {step_count+1} is not uint8. Attempting render. Orig type: {type(next_state_img)}")
                    next_state_img = env.render()
                    if not (isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8):
                        print(
                            f"Error: env.render() for next_state in rgb_array mode did not return a uint8 numpy array for ep {episode_idx+1}, step {step_count+1}. State: {next_state_img}. Skipping step.")
                        step_count += 1
                        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
                            break
                        else:
                            continue
                else:
                    print(
                        f"Warning: Next state obs for ep {episode_idx+1}, step {step_count+1} is not uint8 and env.render_mode is '{env.render_mode}'. Cannot process as image. Skipping step. Obs: {next_state_img}")
                    step_count += 1
                    if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
                        break
                    else:
                        continue

            # Ensure both current and next states are valid image-like arrays after potential rendering
            if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2 and
                    isinstance(next_state_img, np.ndarray) and next_state_img.ndim >= 2):
                print(f"Warning: Skipping step in episode {episode_idx+1}, step {step_count+1} due to unsuitable state dimensions after potential render. Current shape: {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}, Next shape: {next_state_img.shape if hasattr(next_state_img, 'shape') else 'N/A'}")
                # Try to recover with next_state_img if it's valid
                current_state_img = next_state_img
                step_count += 1
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
                    break  # If new current_state_img is also bad
                else:
                    # Try next step with the (potentially problematic) next_state_img as current
                    continue

            # Process the next state frame
            next_frame_pil = preprocess(next_state_img)
            # Convert PIL to tensor without normalization (keep 0-255 range)
            next_frame = torch.from_numpy(np.array(next_frame_pil, dtype=np.uint8)).float()
            # Ensure CHW format
            if len(next_frame.shape) == 2:
                next_frame = next_frame.unsqueeze(0)  # Add channel dimension
            elif len(next_frame.shape) == 3:
                next_frame = next_frame.permute(2, 0, 1)  # HWC to CHW

            # Add the next frame to buffer
            frame_buffer.add_frame(next_frame)
            next_stacked_state = frame_buffer.get_stacked_frames()

            cumulative_reward_episode += accumulated_reward  # Add accumulated reward to cumulative reward for the episode
            
            # Record transition: stacked current state -> action -> single next frame
            episode_transitions.append(
                (recorded_current_stacked_state, action, accumulated_reward, next_stacked_state))
            
            # update current state for next iteration
            current_stacked_state = next_stacked_state

            # Update current state for next iteration
            current_state_img = next_state_img
            step_count += 1 # Increment step_count for the primary transition that was just recorded

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Cumulative Reward: {cumulative_reward_episode:.2f}. Collected {len(episode_transitions)} transitions.")

    env.close()

    if not all_episodes_raw_data:
        print("No data collected. Returning empty datasets.")
        # Pass config to ExperienceDataset constructor
        empty_dataset = ExperienceDataset([], [], [], [], transform=preprocess, config=config)
        return empty_dataset, empty_dataset

    random.shuffle(all_episodes_raw_data)

    num_total_episodes = len(all_episodes_raw_data)
    # Index for end of training set
    split_idx = int((1.0 - validation_split_ratio) * num_total_episodes)

    train_episodes_list = all_episodes_raw_data[:split_idx]
    val_episodes_list = all_episodes_raw_data[split_idx:]

    print(f"Total episodes collected: {num_total_episodes}")
    print(
        f"Splitting into {len(train_episodes_list)} training episodes and {len(val_episodes_list)} validation episodes.")

    # Modified to accept and pass config
    # Note: transform_fn is not used since frames are already preprocessed and stacked
    def create_dataset_from_episode_list(episode_list, transform_fn, dataset_config):
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                # s and ns are already preprocessed tensors (stacked frames)
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)

        # Pass None for transform since data is already preprocessed
        # Pass dataset_config to ExperienceDataset constructor
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=None, config=dataset_config)

    # Pass config to create_dataset_from_episode_list
    train_dataset = create_dataset_from_episode_list(
        train_episodes_list, preprocess, config)
    validation_dataset = create_dataset_from_episode_list(
        val_episodes_list, preprocess, config)

    print(f"Training dataset: {len(train_dataset)} transitions.")
    print(f"Validation dataset: {len(validation_dataset)} transitions.")

    # Save the collected dataset if new data was collected
    if not data_loaded_successfully:
        if all_episodes_raw_data: # Check if new data was actually collected
            num_episodes_collected = config['data']['collection']['num_episodes'] # Or len(all_episodes_raw_data) if actual count is preferred

            # Use config['data']['dataset']['filename'] for saving
            save_path = os.path.join(dataset_dir, save_dataset_filename)

            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': num_episodes_collected,
                'image_height': image_size[0],  # Extract height from tuple
                'image_width': image_size[1],   # Extract width from tuple
                'max_steps_per_episode': max_steps_per_episode,
                'validation_split_ratio': validation_split_ratio,
                'num_train_transitions': len(train_dataset),
                'num_val_transitions': len(validation_dataset),
                'collection_method': 'random', # Added collection method
                'frame_skipping': frame_skipping # Add frame_skipping to metadata
            }

            data_to_save = {
                'train_dataset': train_dataset,
                'val_dataset': validation_dataset,
                'metadata': metadata_to_save
            }

            try:
                torch.save(data_to_save, save_path)
                print(f"Dataset saved to {save_path}")
            except Exception as e:
                print(f"Error saving dataset to {save_path}: {e}")
        else:
            print("No new data was collected, so dataset will not be saved.")

    # Perform distribution check on the train/val split
    if len(train_dataset) > 0 and len(validation_dataset) > 0:
        print("\n" + "="*50)
        print("PERFORMING TRAIN/VAL DISTRIBUTION CHECK")
        print("="*50)
        check_validation_distribution(
            train_dataset, 
            validation_dataset, 
            config=config,
            n_samples=min(500, len(train_dataset), len(validation_dataset)),
            save_plots=config.get('data', {}).get('save_distribution_plots', False),
            plot_dir=config.get('data', {}).get('distribution_plot_dir', "validation_plots")
        )

    return train_dataset, validation_dataset


def collect_ppo_episodes(config, max_steps_per_episode, image_size, validation_split_ratio, frame_skipping):
    env_name = config['environment']['name']
    num_episodes = config['data']['collection']['num_episodes']

    dataset_dir = config['data']['dataset']['dir']
    load_dataset_filename = config['data']['dataset']['load_path']
    save_dataset_filename = config['data']['dataset']['filename']

    os.makedirs(dataset_dir, exist_ok=True)

    data_loaded_successfully = False
    if load_dataset_filename:
        dataset_path = os.path.join(dataset_dir, load_dataset_filename)
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}...")
            try:
                data = torch.load(dataset_path, weights_only=False)
                loaded_train_dataset = data['train_dataset']
                loaded_val_dataset = data['val_dataset']
                metadata = data['metadata']
                print(f"Loaded PPO metadata: {metadata}") # Print loaded metadata
                loaded_frame_skipping = metadata.get('frame_skipping', 'N/A')
                print(f"Frame skipping from loaded PPO metadata: {loaded_frame_skipping}")
                loaded_env_name = metadata.get('environment_name')
                if loaded_env_name != env_name:
                    print(f"Error: Mismatch between loaded dataset environment ('{loaded_env_name}') and config environment ('{env_name}').")
                    raise ValueError("Environment mismatch in loaded dataset.")
                # Check if collection method was PPO, if so, it's likely compatible or what user wants
                if metadata.get('collection_method') == 'ppo':
                    print(f"Successfully loaded PPO-collected dataset for environment '{loaded_env_name}' with {metadata.get('num_episodes_collected', 'N/A')} episodes from {dataset_path}.")
                    data_loaded_successfully = True

                    check_validation_distribution(
                        loaded_train_dataset,
                        loaded_val_dataset,
                        config=config,
                        n_samples=min(500, len(loaded_train_dataset), len(loaded_val_dataset)),
                        save_plots=config.get('data', {}).get('save_distribution_plots', False),
                        plot_dir=config.get('data', {}).get('distribution_plot_dir', "validation_plots")
                    )
                    return loaded_train_dataset, loaded_val_dataset
                else:
                    print(f"Warning: Loaded dataset from {dataset_path} was not collected using PPO (method: {metadata.get('collection_method', 'unknown')}). Proceeding to collect new data with PPO.")
            except Exception as e:
                print(f"Error loading dataset from {dataset_path}: {e}. Proceeding to PPO data collection.")
        else:
            print(f"Warning: Dataset {dataset_path} not found. Proceeding to PPO data collection.")
    else:
        print("`data.dataset.load_path` is empty. Proceeding to PPO data collection.")


    print(f"Collecting data from environment: {env_name} using PPO agent.")
    env_render_mode = None  # Track the successful render mode for parallel envs
    
    # Get image configuration early for environment setup
    env_config = config.get('environment', {})
    image_height = env_config.get('image_height', 64)
    image_width = env_config.get('image_width', 64)
    target_image_size = (image_height, image_width)
    
    try:
        env = gym.make(env_name, render_mode='rgb_array')
        env_render_mode = 'rgb_array'
        print(f"Successfully created env '{env_name}' with render_mode='rgb_array' for PPO collection.")
    except Exception as e_rgb:
        print(f"Failed to create env '{env_name}' with render_mode='rgb_array': {e_rgb}. Trying with render_mode=None...")
        try:
            env = gym.make(env_name, render_mode=None)
            env_render_mode = None
            print(f"Successfully created env '{env_name}' with render_mode=None for PPO collection.")
        except Exception as e_none:
            print(f"Failed to create env '{env_name}' with render_mode=None: {e_none}. Trying without render_mode arg...")
            env = gym.make(env_name)
            env_render_mode = 'default'
            print(f"Successfully created env '{env_name}' with default render_mode ('{env.render_mode if hasattr(env, 'render_mode') else 'unknown'}') for PPO collection.")

    # Apply preprocessing wrappers to the main environment BEFORE PPO training
    print("Applying image preprocessing wrappers for PPO training...")
    env = apply_preprocessing_wrappers(env, config, target_image_size)

    # PPO Agent Setup
    ppo_specific_config = config.get('ppo_agent', {})
    if not ppo_specific_config or not ppo_specific_config.get('enabled', False):
        print("PPO agent configuration is missing or disabled in config. Cannot collect PPO episodes.")
        # Return empty datasets
        empty_dataset = ExperienceDataset(
            [], [], [], [],
            transform=None,  # No transform needed as env is already preprocessed
            config=config
        )
        return empty_dataset, empty_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device  # For Apple Silicon Macs
    print(f"Using device: {device} for PPO agent.")

    # Create a temporary env for PPO training if needed, or use the main `env`
    # It's better to pass the same env instance to ensure compatibility of obs/action spaces

    # Action Repetition Wrapper applied after preprocessing
    action_repetition_k = ppo_specific_config.get('action_repetition_k', 0)
    original_frame_skipping = frame_skipping # Store original frame_skipping

    if action_repetition_k > 1: # k=1 means no repetition, k=0 means not set or disabled
        print(f"Applying ActionRepeatWrapper with k={action_repetition_k}")
        env = ActionRepeatWrapper(env, action_repetition_k)
        frame_skipping = 0 # Disable internal frame skipping if action repetition wrapper is active
        print("Frame skipping disabled due to ActionRepeatWrapper being active.")
    elif action_repetition_k == 1:
        print("action_repetition_k is 1, no ActionRepeatWrapper will be applied.")

    # Create parallel VecEnv for PPO agent training
    # Get number of parallel environments from config (default to CPU count)
    n_envs = ppo_specific_config.get('n_envs', multiprocessing.cpu_count())
    n_envs = max(1, min(n_envs, multiprocessing.cpu_count()))  # Ensure valid range

    
    print(f"Creating {n_envs} parallel environments for PPO training (CPU cores: {multiprocessing.cpu_count()})")
    
    # For debugging, always use DummyVecEnv to avoid subprocess issues
    # Create the environments 
    env_fns = [create_env_for_ppo(env_name, action_repetition_k, env_render_mode, config, target_image_size) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    print(f"Using DummyVecEnv with {n_envs} parallel environments (debugging mode)")

    # Apply VecTransposeImage wrapper to transpose observations from HWC to CHW for CNN policies
    vec_env = VecTransposeImage(vec_env)
    print("Applied VecTransposeImage wrapper to transpose observations from HWC to CHW")


    print(f'Observation space: {vec_env.observation_space}')
    ppo_agent = create_ppo_agent(vec_env, ppo_specific_config, device=device) # Pass vec_env
    train_ppo_agent(ppo_agent, ppo_specific_config, task_name="Initial PPO Training for Data Collection") # Uses the vec_env

    additional_noise = ppo_specific_config.get('additional_log_std_noise', 0.0)
    if additional_noise != 0.0: # Only proceed if noise is non-zero
        if hasattr(ppo_agent.policy, 'log_std') and isinstance(ppo_agent.policy.log_std, torch.Tensor):
            current_log_std = ppo_agent.policy.log_std.data
            noise_tensor = torch.tensor(additional_noise, device=current_log_std.device, dtype=current_log_std.dtype)
            ppo_agent.policy.log_std.data += noise_tensor
            print(f"Adjusted PPO policy log_std by {additional_noise:.4f}")
        elif hasattr(ppo_agent.policy, 'action_dist') and hasattr(ppo_agent.policy.action_dist, 'log_std_param'): # For SquashedGaussian
            current_log_std_param = ppo_agent.policy.action_dist.log_std_param
            if isinstance(current_log_std_param, torch.Tensor):
                noise_tensor = torch.tensor(additional_noise, device=current_log_std_param.device, dtype=current_log_std_param.dtype)
                # For nn.Parameters, modification should be in-place on .data or via an optimizer step if it were training
                current_log_std_param.data += noise_tensor
                print(f"Adjusted PPO policy action_dist.log_std_param by {additional_noise:.4f}")
            else:
                print(f"Warning: PPO policy action_dist.log_std_param found but is not a Tensor. Type: {type(ppo_agent.policy.action_dist.log_std_param)}. Skipping noise addition.")
        else:
            print("Warning: PPO policy does not have a 'log_std' Tensor or 'action_dist.log_std_param' Tensor. Skipping noise addition to log_std.")

    # After training, use the vectorized environment for data collection to maintain consistency
    # Since the PPO agent was trained on vec_env, we should collect data using the same format
    all_episodes_raw_data = []
    
    # Since environment is vectorized and already preprocessed, we need to handle single episode collection
    # Get frame stacking configuration for compatibility checking
    env_config = config.get('environment', {})
    frame_stack_size = env_config.get('frame_stack_size', 1)
    
    print(f"Environment is vectorized and preprocessed. Frame stack size: {frame_stack_size}")
    
    if action_repetition_k > 1:
        print(f"Note: Action repetition is active with k={action_repetition_k}.\n\
              Even though max_steps_per_episode is set to {max_steps_per_episode},\n\
              the actual number of steps per episode may be higher due to action repetition.\n\
              For car_racing_v3, the max_steps_per_episode is typically 1000, but with action repetition it will be {1000 // action_repetition_k} effective steps.\n")

    for episode_idx in range(num_episodes):
        current_state = vec_env.reset()  # Returns batched observations [batch_size, ...]
        
        # Extract the first environment's state (since we're collecting one episode at a time)
        if isinstance(current_state, tuple):
            current_state = current_state[0]  # Handle (obs, info) tuple from newer gym versions
        current_state = current_state[0]  # Get first environment's observation from batch
        
        # VecTransposeImage already converted to CHW format
        if not isinstance(current_state, np.ndarray):
            print(f"Error: Expected numpy array from vectorized env, got {type(current_state)}. Skipping episode {episode_idx+1}.")
            continue

        episode_transitions = []
        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            # Record the current preprocessed state (already in CHW format from VecTransposeImage)
            recorded_current_state = current_state.copy()
            accumulated_reward = 0.0

            # Action selection by PPO agent using preprocessed state
            # For vectorized env, we need to expand state to batch dimension if needed
            if current_state.ndim == 3:  # Single environment state (CHW)
                state_batch = np.expand_dims(current_state, axis=0)  # Add batch dimension
            else:  # Already batched
                state_batch = current_state
                
            action_batch, _ = ppo_agent.predict(state_batch, deterministic=False)
            print(f"Selected action: {action_batch}")
            
            # action_batch is an array of actions for the batch
            # For single environment, extract the scalar action
            if isinstance(action_batch, np.ndarray) and action_batch.shape == (1,):
                original_action = action_batch[0]
            else:
                original_action = action_batch

            # Pass action to vectorized environment
            step_result = vec_env.step(action_batch)
            
            if len(step_result) == 4:
                # Old format: (obs, reward, done, info)
                next_state_batch, reward_batch, done_batch, info_batch = step_result
                # Split done into terminated and truncated
                terminated_batch = done_batch
                truncated_batch = np.zeros_like(done_batch, dtype=bool)
            elif len(step_result) == 5:
                # New format: (obs, reward, terminated, truncated, info)
                next_state_batch, reward_batch, terminated_batch, truncated_batch, info_batch = step_result
            else:
                raise ValueError(f"Unexpected step result length: {len(step_result)}")
            # Extract single environment results from batch
            next_state = next_state_batch[0]
            reward = reward_batch[0]
            terminated = terminated_batch[0]
            truncated = truncated_batch[0]
            # info = info_batch[0]  # Not used, so commented out
            
            accumulated_reward += reward
            current_step_next_state = next_state

            # Frame skipping with vectorized environment
            if frame_skipping > 0:
                for _ in range(frame_skipping):
                    if terminated or truncated:
                        break

                    step_count += 1
                    if step_count >= max_steps_per_episode:
                        break

                    # Action for skip step using preprocessed state
                    if current_step_next_state.ndim == 3:  # Single environment state
                        state_skip_batch = np.expand_dims(current_step_next_state, axis=0)
                    else:  # Already batched
                        state_skip_batch = current_step_next_state
                        
                    action_skip_batch, _ = ppo_agent.predict(state_skip_batch, deterministic=True)
                    step_skip_result = vec_env.step(action_skip_batch)
                    
                    if len(step_skip_result) == 4:
                        # Old format: (obs, reward, done, info)
                        next_state_skip_batch, reward_skip_batch, done_batch, info_batch = step_skip_result
                        terminated_batch = done_batch
                        truncated_batch = np.zeros_like(done_batch, dtype=bool)
                    elif len(step_skip_result) == 5:
                        # New format: (obs, reward, terminated, truncated, info)
                        next_state_skip_batch, reward_skip_batch, terminated_batch, truncated_batch, info_batch = step_skip_result
                    else:
                        raise ValueError(f"Unexpected step skip result length: {len(step_skip_result)}")
                    
                    # Extract results
                    reward_skip = reward_skip_batch[0]
                    terminated = terminated_batch[0]
                    truncated = truncated_batch[0]
                    current_step_next_state = next_state_skip_batch[0]
                    
                    accumulated_reward += reward_skip

            next_state = current_step_next_state

            # Validate preprocessed state dimensions
            if not isinstance(next_state, np.ndarray):
                print(f"Error: Expected numpy array from vectorized env for next state, got {type(next_state)}. Skipping step.")
                step_count += 1
                continue
                
            cumulative_reward_episode += accumulated_reward

            # States are from VecTransposeImage in CHW format as uint8 [0, 255]
            # Keep values in [0, 255] range as requested, just convert to float32 tensors
            current_state_tensor = torch.from_numpy(recorded_current_state).float()

            next_state_tensor = torch.from_numpy(next_state).float()
            
            # Process action for storage
            processed_action = original_action.item() if isinstance(original_action, np.ndarray) and vec_env.action_space.shape == () else original_action
            
            # Record transition: preprocessed current state -> action -> preprocessed next state
            episode_transitions.append((current_state_tensor, processed_action, accumulated_reward, next_state_tensor))
            
            # Update current state for next iteration
            current_state = next_state
            step_count += 1

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"PPO Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Cumulative Reward: {cumulative_reward_episode:.2f}. Collected {len(episode_transitions)} transitions.")

    vec_env.close()  # Close the vectorized environment used for PPO collection

    if not all_episodes_raw_data:
        print("No data collected with PPO. Returning empty datasets.")
        # Pass config to ExperienceDataset constructor
        empty_dataset = ExperienceDataset([], [], [], [], transform=None, config=config)
        return empty_dataset, empty_dataset

    random.shuffle(all_episodes_raw_data)
    num_total_episodes = len(all_episodes_raw_data)
    split_idx = int((1.0 - validation_split_ratio) * num_total_episodes)
    train_episodes_list = all_episodes_raw_data[:split_idx]
    val_episodes_list = all_episodes_raw_data[split_idx:]

    print(f"Total PPO episodes collected: {num_total_episodes}")
    print(f"Splitting into {len(train_episodes_list)} training episodes and {len(val_episodes_list)} validation episodes.")

    # This is a re-definition of the helper function.
    # Since data is already preprocessed, we don't need transform_fn
    def create_dataset_from_episode_list(episode_list, transform_fn, dataset_config):
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                # s and ns are already preprocessed tensors from the wrappers
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)
        # Pass None for transform since data is already preprocessed
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=None, config=dataset_config)

    # Create datasets (transform_fn parameter is not used)
    train_dataset = create_dataset_from_episode_list(train_episodes_list, None, config)
    validation_dataset = create_dataset_from_episode_list(val_episodes_list, None, config)

    print(f"PPO Training dataset: {len(train_dataset)} transitions.")
    print(f"PPO Validation dataset: {len(validation_dataset)} transitions.")

    if not data_loaded_successfully:
        if all_episodes_raw_data:
            num_episodes_collected = config['data']['collection']['num_episodes']
            save_path = os.path.join(dataset_dir, save_dataset_filename)
            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': num_episodes_collected,
                'image_height': target_image_size[0],  # Extract height from tuple
                'image_width': target_image_size[1],   # Extract width from tuple
                'max_steps_per_episode': max_steps_per_episode,
                'validation_split_ratio': validation_split_ratio,
                'num_train_transitions': len(train_dataset),
                'num_val_transitions': len(validation_dataset),
                'collection_method': 'ppo', # Added collection method
                'ppo_config_params': ppo_specific_config, # Save PPO params used for collection
                'frame_skipping': original_frame_skipping, # Save original frame_skipping
                'action_repetition_k': action_repetition_k # Save action_repetition_k
            }
            data_to_save = {
                'train_dataset': train_dataset,
                'val_dataset': validation_dataset,
                'metadata': metadata_to_save
            }
            try:
                torch.save(data_to_save, save_path)
                print(f"PPO collected dataset saved to {save_path}")
            except Exception as e:
                print(f"Error saving PPO dataset to {save_path}: {e}")
        else:
            print("No new PPO data was collected, so dataset will not be saved.")

    # Perform distribution check on the train/val split
    #if len(train_dataset) > 0 and len(validation_dataset) > 0:
    #    check_validation_distribution(
    #        train_dataset, 
    #        validation_dataset, 
    #        config=config,
    #        n_samples=min(500, len(train_dataset), len(validation_dataset)),
    #        save_plots=config.get('data', {}).get('save_distribution_plots', False),
    #        plot_dir=config.get('data', {}).get('distribution_plot_dir', "validation_plots")
    #    )

    return train_dataset, validation_dataset


def check_validation_distribution(train_dataset, val_dataset, config=None, n_samples=1000, 
                                save_plots=False, plot_dir="validation_plots"):
    """
    Check if validation data is in distribution with respect to training data for image datasets.
    Uses basic statistical tests and optionally advanced visualization if dependencies available.
    
    Args:
        train_dataset: Training dataset (ExperienceDataset)
        val_dataset: Validation dataset (ExperienceDataset)
        config: Configuration dictionary (optional)
        n_samples: Number of samples to use for distribution check (for efficiency)
        save_plots: Whether to save distribution plots (requires matplotlib and sklearn)
        plot_dir: Directory to save plots
        
    Returns:
        dict: Dictionary containing distribution similarity metrics and test results
    """
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Warning: One or both datasets are empty. Cannot perform distribution check.")
        return {"status": "error", "message": "Empty datasets"}
    
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available. Using basic statistical comparisons only.")
    
    # Sample data for efficiency if datasets are large
    train_size = min(len(train_dataset), n_samples)
    val_size = min(len(val_dataset), n_samples)
    
    train_indices = random.sample(range(len(train_dataset)), train_size)
    val_indices = random.sample(range(len(val_dataset)), val_size)
    
    print(f"Checking distribution similarity using {train_size} training and {val_size} validation samples...")
    
    # Extract features from images
    train_features = []
    val_features = []
    
    # Extract pixel statistics and basic features
    for idx in train_indices:
        state, action, reward, next_state = train_dataset[idx]
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
        else:
            state_np = np.array(state)
        
        # Flatten image and compute basic statistics
        state_flat = state_np.flatten()
        features = [
            np.mean(state_flat),
            np.std(state_flat),
            np.median(state_flat),
            np.min(state_flat),
            np.max(state_flat),
            np.percentile(state_flat, 25),
            np.percentile(state_flat, 75),
            reward.item() if isinstance(reward, torch.Tensor) else reward
        ]
        train_features.append(features)
    
    for idx in val_indices:
        state, action, reward, next_state = val_dataset[idx]
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
        else:
            state_np = np.array(state)
        
        # Flatten image and compute basic statistics
        state_flat = state_np.flatten()
        features = [
            np.mean(state_flat),
            np.std(state_flat),
            np.median(state_flat),
            np.min(state_flat),
            np.max(state_flat),
            np.percentile(state_flat, 25),
            np.percentile(state_flat, 75),
            reward.item() if isinstance(reward, torch.Tensor) else reward
        ]
        val_features.append(features)
    
    train_features = np.array(train_features)
    val_features = np.array(val_features)
    
    # Statistical tests for each feature
    results = {}
    feature_names = ['mean_pixel', 'std_pixel', 'median_pixel', 'min_pixel', 'max_pixel', 
                    'q25_pixel', 'q75_pixel', 'reward']
    
    for i, feature_name in enumerate(feature_names):
        train_feature = train_features[:, i]
        val_feature = val_features[:, i]
        
        # Basic statistical comparisons
        mean_diff = abs(np.mean(train_feature) - np.mean(val_feature))
        std_diff = abs(np.std(train_feature) - np.std(val_feature))
        
        # Normalized differences (to handle different scales)
        mean_train = np.mean(train_feature)
        normalized_mean_diff = mean_diff / (abs(mean_train) + 1e-8)
        normalized_std_diff = std_diff / (np.std(train_feature) + 1e-8)
        
        result = {
            'train_mean': np.mean(train_feature),
            'val_mean': np.mean(val_feature),
            'train_std': np.std(train_feature),
            'val_std': np.std(val_feature),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'normalized_mean_diff': normalized_mean_diff,
            'normalized_std_diff': normalized_std_diff
        }
        
        # Advanced statistical tests if scipy is available
        if SCIPY_AVAILABLE:
            try:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.ks_2samp(train_feature, val_feature)
                result['ks_statistic'] = ks_stat
                result['ks_p_value'] = ks_p
                
                # Mann-Whitney U test (non-parametric)
                mw_stat, mw_p = stats.mannwhitneyu(train_feature, val_feature, alternative='two-sided')
                result['mw_statistic'] = mw_stat
                result['mw_p_value'] = mw_p
                
                # Anderson-Darling test if samples are large enough
                if len(train_feature) > 25 and len(val_feature) > 25:
                    try:
                        # Use Anderson-Darling 2-sample test if available
                        ad_stat = stats.anderson_ksamp([train_feature, val_feature])
                        result['ad_statistic'] = ad_stat.statistic
                        result['ad_p_value'] = getattr(ad_stat, 'significance_level', None)
                    except Exception as e:
                        print(f"Anderson-Darling test failed for {feature_name}: {e}")
                        result['ad_statistic'] = None
                        result['ad_p_value'] = None
                else:
                    result['ad_statistic'] = None
                    result['ad_p_value'] = None
                    
            except Exception as e:
                print(f"Statistical tests failed for {feature_name}: {e}")
        
        results[feature_name] = result
    
    # Overall assessment
    significant_features = []
    alpha = 0.05  # Significance level
    
    for feature_name, test_results in results.items():
        is_significant = False
        
        if SCIPY_AVAILABLE and 'ks_p_value' in test_results and 'mw_p_value' in test_results:
            # Use statistical tests if available
            if test_results['ks_p_value'] < alpha or test_results['mw_p_value'] < alpha:
                is_significant = True
        else:
            # Use heuristic thresholds for basic comparisons
            if (test_results['normalized_mean_diff'] > 0.1 or 
                test_results['normalized_std_diff'] > 0.2):
                is_significant = True
        
        if is_significant:
            significant_features.append(feature_name)
    
    # Dimensionality reduction for visualization
    if save_plots and MATPLOTLIB_AVAILABLE and SKLEARN_AVAILABLE:
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # PCA visualization
            all_features = np.vstack([train_features, val_features])
            labels = np.concatenate([np.zeros(len(train_features)), np.ones(len(val_features))])
            
            if all_features.shape[1] > 2:
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(all_features)
                
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                train_mask = labels == 0
                val_mask = labels == 1
                plt.scatter(features_2d[train_mask, 0], features_2d[train_mask, 1], 
                           alpha=0.6, label='Training', s=20)
                plt.scatter(features_2d[val_mask, 0], features_2d[val_mask, 1], 
                           alpha=0.6, label='Validation', s=20)
                plt.xlabel(f'PCA Component 1 (var: {pca.explained_variance_ratio_[0]:.2f})')
                plt.ylabel(f'PCA Component 2 (var: {pca.explained_variance_ratio_[1]:.2f})')
                plt.title('PCA: Training vs Validation Distribution')
                plt.legend()
                
                # t-SNE visualization (if not too many samples)
                if len(all_features) <= 1000:
                    plt.subplot(1, 2, 2)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)//4))
                        features_tsne = tsne.fit_transform(all_features)
                    
                    plt.scatter(features_tsne[train_mask, 0], features_tsne[train_mask, 1], 
                               alpha=0.6, label='Training', s=20)
                    plt.scatter(features_tsne[val_mask, 0], features_tsne[val_mask, 1], 
                               alpha=0.6, label='Validation', s=20)
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.title('t-SNE: Training vs Validation Distribution')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not create distribution plots: {e}")
    elif save_plots:
        print("Warning: Plotting disabled due to missing dependencies (matplotlib and/or sklearn)")
    
    # Summary
    in_distribution = len(significant_features) == 0
    
    summary = {
        'in_distribution': in_distribution,
        'significant_features': significant_features,
        'num_significant': len(significant_features),
        'total_features': len(feature_names),
        'train_samples': train_size,
        'val_samples': val_size,
        'feature_tests': results,
        'dependencies': {
            'scipy_available': SCIPY_AVAILABLE,
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE
        }
    }
    
    # Print summary
    print("\n=== Validation Distribution Check Results ===")
    print(f"Training samples analyzed: {train_size}")
    print(f"Validation samples analyzed: {val_size}")
    print(f"Features with significant distribution differences: {len(significant_features)}/{len(feature_names)}")
    
    if significant_features:
        print(f"Significantly different features: {', '.join(significant_features)}")
        print("  WARNING: Validation set may not be representative of training distribution!")
    else:
        print(" Validation set appears to be in distribution with training set")
    
    if len(significant_features) > len(feature_names) // 2:
        print(" CRITICAL: More than half of features show significant differences!")
        print("   Consider re-shuffling or collecting more diverse data.")
    
    return summary


def apply_preprocessing_wrappers(env, config, image_size):
    """
    Apply preprocessing wrappers to environment based on config.
    
    Args:
        env: Base gymnasium environment
        config: Configuration dictionary
        image_size: Target image size as (height, width)
        
    Returns:
        Wrapped environment
    """
    env_config = config.get('environment', {})
    
    # Apply image preprocessing wrapper
    grayscale = True
    env = ImagePreprocessingWrapper(env, image_size, grayscale=grayscale)
    
    # Apply frame stacking wrapper
    frame_stack_size = env_config.get('frame_stack_size', 1)
    if frame_stack_size > 1:
        env = FrameStackWrapper(env, frame_stack_size)
    
    return env


def create_env_for_ppo(env_name, action_repetition_k=0, original_render_mode=None, config=None, image_size=None):
    """
    Factory function to create environment instances for parallel PPO training.
    This is needed for SubprocVecEnv which requires a callable that returns an environment.
    
    Args:
        env_name: Name of the gymnasium environment
        action_repetition_k: Action repetition factor (0 or 1 means no repetition)
        original_render_mode: The render mode to use for environment creation
        config: Configuration dictionary for applying preprocessing wrappers
        image_size: Target image size as (height, width)
    
    Returns:
        Callable that returns a gymnasium environment
    """
    def _make_env():
        # Try to create environment with specified render mode
        if original_render_mode == 'rgb_array':
            try:
                env = gym.make(env_name, render_mode='rgb_array')
            except Exception:
                try:
                    env = gym.make(env_name, render_mode=None)
                except Exception:
                    env = gym.make(env_name)
        elif original_render_mode is None:
            try:
                env = gym.make(env_name, render_mode=None)
            except Exception:
                env = gym.make(env_name)
        else:
            try:
                env = gym.make(env_name)
            except Exception:
                env = gym.make(env_name, render_mode=None)
        
        # Apply image preprocessing wrappers if config is provided
        if config is not None and image_size is not None:
            env = apply_preprocessing_wrappers(env, config, image_size)
        
        # Apply action repetition wrapper if needed (after preprocessing)
        if action_repetition_k > 1:
            env = ActionRepeatWrapper(env, action_repetition_k)
        
        return env
    
    return _make_env


if __name__ == '__main__':
    # Example usage:
    # Ensure you have a display server if using environments like CarRacing-v2 locally without headless mode.
    # For servers, use Xvfb: Xvfb :1 -screen 0 1024x768x24 &
    # export DISPLAY=:1
    # The __main__ block below is primarily for testing collect_random_episodes.
    # Testing collect_ppo_episodes would require a more elaborate setup,
    # including a full config with ppo_agent settings.

    print("Testing data collection with a sample environment (collect_random_episodes)...")
    try:
        # Attempt to use a known pixel-based environment
        try:
            # Try a simpler environment first that is less likely to have missing dependencies
            gym.make("CartPole-v1")
            test_env_name = "CartPole-v1"
            print("Using CartPole-v1 for testing data collection (will try to render for image).")
        except gym.error.MissingEnvDependency:
            try:
                gym.make("PongNoFrameskip-v4")
                test_env_name = "PongNoFrameskip-v4"
                print("Using PongNoFrameskip-v4 for testing data collection.")
            except gym.error.MissingEnvDependency:
                print("Neither CartPole-v1 nor PongNoFrameskip-v4 are available. Skipping data_utils.py example run.")
                test_env_name = None
        except Exception as e: # Catch other potential errors during gym.make
            print(f"Could not make test environment: {e}. Skipping data_utils.py example run.")
            test_env_name = None


        if test_env_name:
            # Dummy config for testing collect_random_episodes
            # Add necessary keys for the new validation logic in ExperienceDataset
            dummy_config_random = {
                'environment': {
                    'name': test_env_name,
                    'input_channels': 3, # Assuming RGB for test environments like Pong, CartPole gives 3 channels after render
                    'image_height': 32,   # Must match image_size below for T.Resize and validation
                    'image_width': 32     # Must match image_size below for T.Resize and validation
                },
                'data': {
                    'collection': {
                        'num_episodes': 2 # Small number for test
                    },
                    'dataset': {
                        'dir': "datasets/random_test",
                        'load_path': "", # Don't load, force collection
                        'filename': "random_collected_data.pth"
                    }
                }
            }

            print("\n--- Test Case 1: Collect and Save (Random Episodes) ---")
            train_d, val_d = collect_random_episodes(
                config=dummy_config_random,
                max_steps_per_episode=30,
                image_size=(32, 32), # Smaller images for faster test
                validation_split_ratio=0.5,
                frame_skipping=2
            )

            print(f"\n--- Random Training Dataset (Size: {len(train_d)}) ---")
            if len(train_d) > 0:
                train_dataloader = DataLoader(train_d, batch_size=2, shuffle=True)
                s_batch, a_batch, r_batch, s_next_batch = next(iter(train_dataloader))
                print(f"Random Training Sample batch shapes: States {s_batch.shape}, Actions {a_batch.shape}, Rewards {r_batch.shape}, Next States {s_next_batch.shape}")
            else:
                print("Random training dataset is empty.")

            print(f"\n--- Random Validation Dataset (Size: {len(val_d)}) ---")
            if len(val_d) > 0:
                val_dataloader = DataLoader(val_d, batch_size=2, shuffle=False)
                s_val_batch, a_val_batch, r_val_batch, s_next_val_batch = next(iter(val_dataloader))
                print(f"Random Validation Sample batch shapes: States {s_val_batch.shape}, Actions {a_val_batch.shape}, Rewards {r_val_batch.shape}, Next States {s_next_val_batch.shape}")
            else:
                print("Random validation dataset is empty.")

            # Test case 2: Load the saved random dataset
            print("\n--- Test Case 2: Load Saved Random Dataset ---")
            dummy_config_load_random = {
                'environment': {
                    'name': test_env_name,
                    'input_channels': 3, # Must match for validation when loading
                    'image_height': 32,   # Must match for validation when loading
                    'image_width': 32     # Must match for validation when loading
                },
                'data': {
                    'collection': {
                        'num_episodes': dummy_config_random['data']['collection']['num_episodes']
                    },
                    'dataset': {
                        'dir': dummy_config_random['data']['dataset']['dir'],
                        'load_path': dummy_config_random['data']['dataset']['filename'], # What test case 1 saved
                        'filename': "random_collected_data_new_save.pth" # In case this run also saves
                    }
                }
            }

            dataset_file_path = os.path.join(dummy_config_load_random['data']['dataset']['dir'], dummy_config_load_random['data']['dataset']['load_path'])
            if os.path.exists(dataset_file_path):
                train_d_loaded, val_d_loaded = collect_random_episodes(
                    config=dummy_config_load_random,
                    max_steps_per_episode=30,
                    image_size=(32, 32),
                    validation_split_ratio=0.5,
                    frame_skipping=2
                )

                print(f"\n--- Loaded Random Training Dataset (Size: {len(train_d_loaded)}) ---")
                if len(train_d_loaded) > 0:
                    assert len(train_d_loaded) == len(train_d), "Loaded random train dataset size mismatch!"
                    print("Loaded random training dataset size matches original.")
                else:
                    print("Loaded random training dataset is empty.")

                print(f"\n--- Loaded Random Validation Dataset (Size: {len(val_d_loaded)}) ---")
                if len(val_d_loaded) > 0:
                    assert len(val_d_loaded) == len(val_d), "Loaded random validation dataset size mismatch!"
                    print("Loaded random validation dataset size matches original.")
                else:
                    print("Loaded random validation dataset is empty.")
            else:
                print(f"Random dataset file {dataset_file_path} not found for Test Case 2. Skipping loading test.")

    except ImportError as e:
        print(f"Import error during test, likely missing a dependency for the test environment: {e}")
    except Exception as e:
        print(f"An error occurred during the data_utils.py example run: {e}")
        import traceback
        traceback.print_exc()
