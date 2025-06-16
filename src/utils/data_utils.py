import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random  # Added for shuffling episodes
import os
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.rl_agent import create_ppo_agent, train_ppo_agent
from src.utils.env_wrappers import ActionRepeatWrapper
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

        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)

        if self.config:
            expected_input_channels = self.config['input_channels']
            # image_size in config is an int (e.g., 64)
            expected_image_height = self.config['image_size']
            expected_image_width = self.config['image_size']
            expected_shape = (expected_input_channels, expected_image_height, expected_image_width)

            if state.shape != expected_shape:
                error_msg = f"State shape {state.shape} does not match target {expected_shape} for idx {idx}."
                # print(f"STATE (raw an transformed): {self.states[idx]} --> {state}") # for debugging
                raise ValueError(error_msg)

            if next_state.shape != expected_shape:
                error_msg = f"Next state shape {next_state.shape} does not match target {expected_shape} for idx {idx}."
                # print(f"NEXT_STATE (raw an transformed): {self.next_states[idx]} --> {next_state}") # for debugging
                raise ValueError(error_msg)

        # Convert action to tensor, ensure it's float for potential nn.Linear embedding
        # Adjust dtype based on action type (discrete typically long, continuous float)
        # For simplicity, let's assume actions will be made float.
        # If discrete, they might be indices; ensure they are handled appropriately later (e.g. one-hot or embedding layer).
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        return state, action_tensor, reward_tensor, next_state


def collect_random_episodes(config, max_steps_per_episode, image_size, validation_split_ratio, frame_skipping):
    env_name = config['environment_name']
    num_episodes = config['num_episodes_data_collection']

    # New configuration keys
    dataset_dir = config['dataset_dir']
    load_dataset_filename = config['load_dataset_path'] # Filename or path relative to dataset_dir
    save_dataset_filename = config['dataset_filename'] # Filename for saving new datasets

    os.makedirs(dataset_dir, exist_ok=True)

    data_loaded_successfully = False
    if load_dataset_filename: # If not an empty string, attempt to load
        dataset_path = os.path.join(dataset_dir, load_dataset_filename)
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}...")
            try:
                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f)

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
        print("`load_dataset_path` is empty. Proceeding to data collection.")

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

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])

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

        episode_transitions = []
        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0  # Initialize cumulative reward

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            recorded_current_state_img = current_state_img
            accumulated_reward = 0.0

            action = env.action_space.sample() # Primary action

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

            # The transition uses recorded_current_state_img, the original action, accumulated_reward, and the final next_state_img after skipping
            episode_transitions.append(
                (recorded_current_state_img, action, accumulated_reward, next_state_img))
            current_state_img = next_state_img # Update current state for the next iteration of the main while loop
            step_count += 1 # Increment step_count for the primary transition that was just recorded

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Collected {len(episode_transitions)} transitions.")

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
    def create_dataset_from_episode_list(episode_list, transform_fn, dataset_config):
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)

        # If flat_states is empty, ExperienceDataset will handle it (or should)
        # Pass dataset_config to ExperienceDataset constructor
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=transform_fn, config=dataset_config)

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
            num_episodes_collected = config['num_episodes_data_collection'] # Or len(all_episodes_raw_data) if actual count is preferred

            # Use config['dataset_filename'] for saving
            save_path = os.path.join(dataset_dir, save_dataset_filename)

            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': num_episodes_collected,
                'image_size': image_size,
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
                with open(save_path, 'wb') as f:
                    pickle.dump(data_to_save, f)
                print(f"Dataset saved to {save_path}")
            except Exception as e:
                print(f"Error saving dataset to {save_path}: {e}")
        else:
            print("No new data was collected, so dataset will not be saved.")


    return train_dataset, validation_dataset


def collect_ppo_episodes(config, max_steps_per_episode, image_size, validation_split_ratio, frame_skipping):
    env_name = config['environment_name']
    num_episodes = config['num_episodes_data_collection']

    dataset_dir = config['dataset_dir']
    load_dataset_filename = config['load_dataset_path']
    save_dataset_filename = config['dataset_filename']

    os.makedirs(dataset_dir, exist_ok=True)

    data_loaded_successfully = False
    if load_dataset_filename:
        dataset_path = os.path.join(dataset_dir, load_dataset_filename)
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}...")
            try:
                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f)
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
                    return loaded_train_dataset, loaded_val_dataset
                else:
                    print(f"Warning: Loaded dataset from {dataset_path} was not collected using PPO (method: {metadata.get('collection_method', 'unknown')}). Proceeding to collect new data with PPO.")
            except Exception as e:
                print(f"Error loading dataset from {dataset_path}: {e}. Proceeding to PPO data collection.")
        else:
            print(f"Warning: Dataset {dataset_path} not found. Proceeding to PPO data collection.")
    else:
        print("`load_dataset_path` is empty. Proceeding to PPO data collection.")


    print(f"Collecting data from environment: {env_name} using PPO agent.")
    try:
        env = gym.make(env_name, render_mode='rgb_array')
        print(f"Successfully created env '{env_name}' with render_mode='rgb_array' for PPO collection.")
    except Exception as e_rgb:
        print(f"Failed to create env '{env_name}' with render_mode='rgb_array': {e_rgb}. Trying with render_mode=None...")
        try:
            env = gym.make(env_name, render_mode=None)
            print(f"Successfully created env '{env_name}' with render_mode=None for PPO collection.")
        except Exception as e_none:
            print(f"Failed to create env '{env_name}' with render_mode=None: {e_none}. Trying without render_mode arg...")
            env = gym.make(env_name)
            print(f"Successfully created env '{env_name}' with default render_mode ('{env.render_mode if hasattr(env, 'render_mode') else 'unknown'}') for PPO collection.")

    # PPO Agent Setup
    ppo_specific_config = config.get('ppo_agent', {})
    if not ppo_specific_config or not ppo_specific_config.get('enabled', False):
        print("PPO agent configuration is missing or disabled in config. Cannot collect PPO episodes.")
        # Return empty datasets or raise an error
        # Pass config to ExperienceDataset constructor
        empty_dataset = ExperienceDataset(
            [], [], [], [],
            transform=T.Compose([T.ToPILImage(),T.Resize(image_size),T.ToTensor()]),
            config=config
        )
        return empty_dataset, empty_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device  # For Apple Silicon Macs
    print(f"Using device: {device} for PPO agent.")

    # Create a temporary env for PPO training if needed, or use the main `env`
    # It's better to pass the same env instance to ensure compatibility of obs/action spaces

    # Action Repetition Wrapper
    action_repetition_k = ppo_specific_config.get('action_repetition_k', 0)
    original_frame_skipping = frame_skipping # Store original frame_skipping

    if action_repetition_k > 1: # k=1 means no repetition, k=0 means not set or disabled
        print(f"Applying ActionRepeatWrapper with k={action_repetition_k}")
        env = ActionRepeatWrapper(env, action_repetition_k)
        frame_skipping = 0 # Disable internal frame skipping if action repetition wrapper is active
        print(f"Frame skipping disabled due to ActionRepeatWrapper being active.")
    elif action_repetition_k == 1:
        print("action_repetition_k is 1, no ActionRepeatWrapper will be applied.")

    # Create VecEnv for PPO agent
    vec_env = DummyVecEnv([lambda: env])

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
            print(f"Warning: PPO policy does not have a 'log_std' Tensor or 'action_dist.log_std_param' Tensor. Skipping noise addition to log_std.")

    # After training, the env used by PPO (which is `env` wrapped in DummyVecEnv) should still be usable
    # as SB3 usually doesn't close the original envs passed to DummyVecEnv unless DummyVecEnv.close() is called,
    # which happens if ppo_agent.env.close() is called. `learn()` does not close it.

    all_episodes_raw_data = []
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])

    for episode_idx in range(num_episodes):
        current_state_img, info = env.reset()
        initial_obs_is_uint8_image = isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8

        if not initial_obs_is_uint8_image:
            if env.render_mode == 'rgb_array':
                current_state_img = env.render()
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8):
                    print(f"Error: env.render() did not return a uint8 numpy array for PPO ep {episode_idx+1}. Skipping.")
                    continue
            else:
                print(f"Warning: Initial PPO obs for ep {episode_idx+1} not uint8 and env not in 'rgb_array' mode. Skipping. Obs: {current_state_img}")
                continue

        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
            print(f"Skipping PPO episode {episode_idx+1} due to unsuitable initial state. State: {current_state_img}")
            continue

        episode_transitions = []
        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0  # Initialize cumulative reward

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            recorded_current_state_img = current_state_img
            accumulated_reward = 0.0

            # Action selection by PPO agent for the primary action
            original_action, _ = ppo_agent.predict(recorded_current_state_img, deterministic=True)

            # First step
            next_state_img, reward, terminated, truncated, info = env.step(original_action)
            accumulated_reward += reward
            current_step_next_state_img = next_state_img
            # Similar to random collection, step_count for the main recorded transition is incremented after skipping.

            if frame_skipping > 0:
                for _ in range(frame_skipping):
                    if terminated or truncated:
                        break

                    step_count += 1 # This step is part of the episode's max_steps
                    if step_count >= max_steps_per_episode:
                        break

                    # Action for skip step based on the *actual current* state in the environment
                    action_skip, _ = ppo_agent.predict(current_step_next_state_img, deterministic=True)
                    next_state_skip, reward_skip, terminated, truncated, info = env.step(action_skip)
                    accumulated_reward += reward_skip
                    current_step_next_state_img = next_state_skip

            next_state_img = current_step_next_state_img # S' after all skips

            # Standardize image check for next_state_img (which is current_step_next_state_img)
            next_obs_is_uint8_image = isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8

            if not next_obs_is_uint8_image:
                if env.render_mode == 'rgb_array':
                    next_state_img = env.render()
                    if not (isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8):
                        print(f"Error: env.render() for next_state (PPO) did not return uint8 array for ep {episode_idx+1}, step {step_count+1}. Skipping step.")
                        step_count += 1
                        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2): break
                        else: continue
                else:
                    print(f"Warning: Next PPO obs for ep {episode_idx+1}, step {step_count+1} not uint8 and env not 'rgb_array'. Skipping step. Obs: {next_state_img}")
                    step_count += 1
                    if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2): break
                    else: continue
                
            cumulative_reward_episode += accumulated_reward # Add accumulated reward to cumulative reward for the episode

            if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2 and
                    isinstance(next_state_img, np.ndarray) and next_state_img.ndim >= 2):
                print(f"Warning: Skipping PPO step in ep {episode_idx+1}, step {step_count+1} due to unsuitable state dims. Current: {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}, Next: {next_state_img.shape if hasattr(next_state_img, 'shape') else 'N/A'}")
                current_state_img = next_state_img
                step_count += 1
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2): break
                else: continue

            # The transition uses recorded_current_state_img, the original_action, accumulated_reward, and the final next_state_img
            processed_action = original_action.item() if isinstance(original_action, np.ndarray) and env.action_space.shape == () else original_action
            episode_transitions.append((recorded_current_state_img, processed_action, accumulated_reward, next_state_img))
            current_state_img = next_state_img # Update current state for the next iteration
            step_count += 1 # Increment step_count for the primary transition

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"PPO Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Cumulative Reward: {cumulative_reward_episode:.2f}. Collected {len(episode_transitions)} transitions.")

    env.close() # Close the environment used for PPO collection

    if not all_episodes_raw_data:
        print("No data collected with PPO. Returning empty datasets.")
        # Pass config to ExperienceDataset constructor
        empty_dataset = ExperienceDataset([], [], [], [], transform=preprocess, config=config)
        return empty_dataset, empty_dataset

    random.shuffle(all_episodes_raw_data)
    num_total_episodes = len(all_episodes_raw_data)
    split_idx = int((1.0 - validation_split_ratio) * num_total_episodes)
    train_episodes_list = all_episodes_raw_data[:split_idx]
    val_episodes_list = all_episodes_raw_data[split_idx:]

    print(f"Total PPO episodes collected: {num_total_episodes}")
    print(f"Splitting into {len(train_episodes_list)} training episodes and {len(val_episodes_list)} validation episodes.")

    # This is a re-definition of the helper function.
    # It should be identical to the one in collect_random_episodes if it's meant to be shared,
    # or named differently if specific. Assuming it's intended to be the same.
    # It already has been modified in the previous hunk for collect_random_episodes if this diff is applied sequentially.
    # For safety, ensuring the change here if it wasn't.
    def create_dataset_from_episode_list(episode_list, transform_fn, dataset_config): # Added dataset_config
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)
        # Pass dataset_config to ExperienceDataset constructor
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=transform_fn, config=dataset_config)

    # Pass config to create_dataset_from_episode_list
    train_dataset = create_dataset_from_episode_list(train_episodes_list, preprocess, config)
    validation_dataset = create_dataset_from_episode_list(val_episodes_list, preprocess, config)

    print(f"PPO Training dataset: {len(train_dataset)} transitions.")
    print(f"PPO Validation dataset: {len(validation_dataset)} transitions.")

    if not data_loaded_successfully:
        if all_episodes_raw_data:
            num_episodes_collected = config['num_episodes_data_collection']
            save_path = os.path.join(dataset_dir, save_dataset_filename)
            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': num_episodes_collected,
                'image_size': image_size,
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
                with open(save_path, 'wb') as f:
                    pickle.dump(data_to_save, f)
                print(f"PPO collected dataset saved to {save_path}")
            except Exception as e:
                print(f"Error saving PPO dataset to {save_path}: {e}")
        else:
            print("No new PPO data was collected, so dataset will not be saved.")

    return train_dataset, validation_dataset


if __name__ == '__main__':
    # Example usage:
    # Ensure you have a display server if using environments like CarRacing-v2 locally without headless mode.
    # For servers, use Xvfb: Xvfb :1 -screen 0 1024x768x24 &
    # export DISPLAY=:1
    # The __main__ block below is primarily for testing collect_random_episodes.
    # Testing collect_ppo_episodes would require a more elaborate setup,
    # including a full config with ppo_agent settings.

    print(f"Testing data collection with a sample environment (collect_random_episodes)...")
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
                'environment_name': test_env_name,
                'num_episodes_data_collection': 2, # Small number for test
                'dataset_dir': "datasets/random_test",
                'load_dataset_path': "", # Don't load, force collection
                'dataset_filename': "random_collected_data.pkl",
                'input_channels': 3, # Assuming RGB for test environments like Pong, CartPole gives 3 channels after render
                'image_size': 32      # Must match image_size below for T.Resize and validation
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
                'environment_name': test_env_name,
                'num_episodes_data_collection': dummy_config_random['num_episodes_data_collection'],
                'dataset_dir': dummy_config_random['dataset_dir'],
                'load_dataset_path': dummy_config_random['dataset_filename'], # What test case 1 saved
                'dataset_filename': "random_collected_data_new_save.pkl", # In case this run also saves
                'input_channels': 3, # Must match for validation when loading
                'image_size': 32      # Must match for validation when loading
            }

            dataset_file_path = os.path.join(dummy_config_load_random['dataset_dir'], dummy_config_load_random['load_dataset_path'])
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
