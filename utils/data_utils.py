import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random  # Added for shuffling episodes
# import cv2 # For image resizing - Removed as torchvision.transforms is used


class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, transform=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.transform = transform

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

        # Convert action to tensor, ensure it's float for potential nn.Linear embedding
        # Adjust dtype based on action type (discrete typically long, continuous float)
        # For simplicity, let's assume actions will be made float.
        # If discrete, they might be indices; ensure they are handled appropriately later (e.g. one-hot or embedding layer).
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        return state, action_tensor, reward_tensor, next_state


def collect_random_episodes(env_name, num_episodes, max_steps_per_episode, image_size, validation_split_ratio):
    # Default image_size removed from signature to match common practice when it's passed from config.
    # Ensure image_size is provided when calling.
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

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            action = env.action_space.sample()
            next_state_img, reward, terminated, truncated, info = env.step(
                action)

            # Standardize image check for next_state
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

            episode_transitions.append(
                (current_state_img, action, reward, next_state_img))
            current_state_img = next_state_img
            step_count += 1

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Collected {len(episode_transitions)} transitions.")

    env.close()

    if not all_episodes_raw_data:
        print("No data collected. Returning empty datasets.")
        empty_dataset = ExperienceDataset([], [], [], [], transform=preprocess)
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

    def create_dataset_from_episode_list(episode_list, transform_fn):
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)

        # If flat_states is empty, ExperienceDataset will handle it (or should)
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=transform_fn)

    train_dataset = create_dataset_from_episode_list(
        train_episodes_list, preprocess)
    validation_dataset = create_dataset_from_episode_list(
        val_episodes_list, preprocess)

    print(f"Training dataset: {len(train_dataset)} transitions.")
    print(f"Validation dataset: {len(validation_dataset)} transitions.")

    return train_dataset, validation_dataset


if __name__ == '__main__':
    # Example usage:
    # Ensure you have a display server if using environments like CarRacing-v2 locally without headless mode.
    # For servers, use Xvfb: Xvfb :1 -screen 0 1024x768x24 &
    # export DISPLAY=:1

    # Test with a simpler environment first if CarRacing-v2 causes issues
    # env_name_test = "CartPole-v1" # This is not pixel based by default
    # Atari, pixel based. Needs AutoROM: pip install gymnasium[accept-rom-license]
    env_name_test = "PongNoFrameskip-v4"
    # env_name_test = "CarRacing-v2" # Needs Box2D

    # To run CarRacing-v2, you might need:
    # sudo apt-get install swig
    # pip install gymnasium[box2d]

    # For Atari games:
    # pip install gymnasium[atari] gymnasium[accept-rom-license]

    print(f"Testing data collection with a sample environment...")
    try:
        # Attempt to use a known pixel-based environment
        # If 'CarRacing-v2' is in config, it will be tried by default by train.py
        # For this standalone test, let's try Pong if available, else a warning.
        try:
            gym.make("PongNoFrameskip-v4")  # Check if env is available
            test_env = "PongNoFrameskip-v4"
            print("Using PongNoFrameskip-v4 for testing data collection.")
        except gym.error.MissingEnvDependency:  # Corrected exception type
            print(
                "PongNoFrameskip-v4 not available (Atari ROMs likely missing or 'gymnasium[accept-rom-license]' not used).")
            print("Skipping data_utils.py example run.")
            test_env = None

        if test_env:
            # Example: Collect 5 episodes, split 60% train / 40% validation
            # image_size must be passed as it's no longer defaulted in the function signature
            train_d, val_d = collect_random_episodes(
                env_name=test_env,
                num_episodes=5,
                max_steps_per_episode=50,
                image_size=(64, 64),  # Explicitly pass image_size
                validation_split_ratio=0.4
            )

            print(f"\n--- Training Dataset (Size: {len(train_d)}) ---")
            if len(train_d) > 0:
                train_dataloader = DataLoader(
                    train_d, batch_size=4, shuffle=True)
                s_batch, a_batch, r_batch, s_next_batch = next(
                    iter(train_dataloader))
                print(f"Training Sample batch shapes:")
                print(
                    f"  States (s_t): {s_batch.shape}, dtype: {s_batch.dtype}")
                print(
                    f"  Actions (a_t): {a_batch.shape}, dtype: {a_batch.dtype}")
                print(
                    f"  Rewards (r_t): {r_batch.shape}, dtype: {r_batch.dtype}")
                print(
                    f"  Next States (s_t+1): {s_next_batch.shape}, dtype: {s_next_batch.dtype}")
            else:
                print("Training dataset is empty.")

            print(f"\n--- Validation Dataset (Size: {len(val_d)}) ---")
            if len(val_d) > 0:
                val_dataloader = DataLoader(val_d, batch_size=4, shuffle=False)
                s_val_batch, a_val_batch, r_val_batch, s_next_val_batch = next(
                    iter(val_dataloader))
                print(f"Validation Sample batch shapes:")
                print(
                    f"  States (s_t): {s_val_batch.shape}, dtype: {s_val_batch.dtype}")
                print(
                    f"  Actions (a_t): {a_val_batch.shape}, dtype: {a_val_batch.dtype}")
                print(
                    f"  Rewards (r_t): {r_val_batch.shape}, dtype: {r_val_batch.dtype}")
                print(
                    f"  Next States (s_t+1): {s_next_val_batch.shape}, dtype: {s_next_val_batch.dtype}")
            else:
                print("Validation dataset is empty.")

    except ImportError as e:
        print(
            f"Import error, likely missing a dependency for the test environment: {e}")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
