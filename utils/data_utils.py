import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
# import cv2 # For image resizing - Removed as torchvision.transforms is used

class ExperienceDataset(Dataset):
    def __init__(self, states, actions, next_states, transform=None):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        next_state = self.next_states[idx]

        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)
        
        # Convert action to tensor, ensure it's float for potential nn.Linear embedding
        # Adjust dtype based on action type (discrete typically long, continuous float)
        # For simplicity, let's assume actions will be made float.
        # If discrete, they might be indices; ensure they are handled appropriately later (e.g. one-hot or embedding layer).
        action_tensor = torch.tensor(action, dtype=torch.float32)

        return state, action_tensor, next_state

def collect_random_episodes(env_name, num_episodes, max_steps_per_episode, image_size=(64, 64)):
    print(f"Collecting data from environment: {env_name}")
    try:
        # Try to make the environment, handling potential render mode issues for headless servers
        env = gym.make(env_name, render_mode=None) # No rendering during data collection
    except Exception as e:
        print(f"Failed to create env with render_mode=None: {e}. Trying with 'rgb_array'...")
        try:
            env = gym.make(env_name, render_mode='rgb_array')
        except Exception as e_rgb:
            print(f"Failed to create env with render_mode='rgb_array': {e_rgb}. Trying without render_mode arg...")
            env = gym.make(env_name) # Fallback

    states_list = []
    actions_list = []
    next_states_list = []

    # Define a basic transform for images: Resize and convert to CHW tensor
    # Note: Normalization might be added later if needed, depending on model performance.
    # For now, just resize and to tensor (which scales to [0,1] if input is uint8).
    preprocess = T.Compose([
        T.ToPILImage(), # Convert numpy array (H,W,C) to PIL Image
        T.Resize(image_size),
        T.ToTensor() # Converts PIL Image (H,W,C) or numpy.ndarray (H,W,C) to (C,H,W) tensor and scales to [0,1]
    ])

    for episode in range(num_episodes):
        current_state_img, info = env.reset()
        if not isinstance(current_state_img, np.ndarray) or current_state_img.dtype != np.uint8:
            print(f"Warning: Initial observation is not a standard image (numpy array, uint8). Type: {type(current_state_img)}, Dtype: {current_state_img.dtype if hasattr(current_state_img, 'dtype') else 'N/A'}")
            # Attempt to get a render if reset() doesn't return image directly
            if hasattr(env, 'render') and 'rgb_array' in env.metadata.get('render_modes', []):
                 current_state_img = env.render()
            elif env.render_mode == 'rgb_array': # If mode was set to rgb_array
                 current_state_img = env.render()


        if not isinstance(current_state_img, np.ndarray) or current_state_img.ndim < 2 : # Simple check
            print(f"Skipping episode {episode+1} due to non-image state from env.reset() or render(). State: {current_state_img}")
            continue


        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            action = env.action_space.sample() # Sample a random action

            next_state_img, reward, terminated, truncated, info = env.step(action)
            
            if not isinstance(next_state_img, np.ndarray) or next_state_img.dtype != np.uint8:
                 if hasattr(env, 'render') and 'rgb_array' in env.metadata.get('render_modes', []):
                    next_state_img = env.render()
                 elif env.render_mode == 'rgb_array':
                    next_state_img = env.render()


            if not isinstance(current_state_img, np.ndarray) or current_state_img.ndim < 2 or \
               not isinstance(next_state_img, np.ndarray) or next_state_img.ndim < 2:
                print(f"Warning: Skipping step due to non-image state. Current: {type(current_state_img)}, Next: {type(next_state_img)}")
                current_state_img = next_state_img # Move to next state hoping it's valid
                step_count += 1
                if isinstance(current_state_img, np.ndarray) and current_state_img.ndim <2 : # if still bad, break
                    break 
                else:
                    continue


            # Store raw numpy arrays first. Transformation will be done by Dataset or here.
            # For ExperienceDataset, it expects PIL images or tensors based on transform.
            # Let's store them as numpy arrays and let the Dataset handle transforms.
            # The transform defined above expects PIL images.
            
            states_list.append(current_state_img) # HWC numpy array
            actions_list.append(action)
            next_states_list.append(next_state_img) # HWC numpy array

            current_state_img = next_state_img
            step_count += 1

        print(f"Episode {episode+1}/{num_episodes} finished after {step_count} steps.")

    env.close()
    
    if not states_list:
        print("No data collected. Check environment compatibility and observation space.")
        # Return empty tensors or raise error
        return torch.empty(0), torch.empty(0), torch.empty(0)


    # Create the dataset
    # The ExperienceDataset will apply the 'preprocess' transform
    dataset = ExperienceDataset(states_list, actions_list, next_states_list, transform=preprocess)
    
    print(f"Collected {len(dataset)} (state, action, next_state) pairs.")
    
    # This function can return the dataset directly, or the raw lists.
    # Returning the dataset is often more convenient.
    return dataset


if __name__ == '__main__':
    # Example usage:
    # Ensure you have a display server if using environments like CarRacing-v2 locally without headless mode.
    # For servers, use Xvfb: Xvfb :1 -screen 0 1024x768x24 &
    # export DISPLAY=:1
    
    # Test with a simpler environment first if CarRacing-v2 causes issues
    # env_name_test = "CartPole-v1" # This is not pixel based by default
    env_name_test = "PongNoFrameskip-v4" # Atari, pixel based. Needs AutoROM: pip install gymnasium[accept-rom-license]
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
            gym.make("PongNoFrameskip-v4") # Check if env is available
            test_env = "PongNoFrameskip-v4"
            print("Using PongNoFrameskip-v4 for testing data collection.")
        except gym.error.MissingEnvDependency:
            print("PongNoFrameskip-v4 not available (Atari ROMs likely missing or 'gymnasium[accept-rom-license]' not used).")
            print("Skipping data_utils.py example run.")
            test_env = None

        if test_env:
            dataset = collect_random_episodes(
                env_name=test_env,
                num_episodes=2, # Small number for testing
                max_steps_per_episode=50,
                image_size=(64, 64)
            )

            if len(dataset) > 0:
                # Create a DataLoader
                dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
                
                # Sample a batch
                s_batch, a_batch, s_next_batch = next(iter(dataloader))
                
                print(f"Sample batch shapes:")
                print(f"States (s_t): {s_batch.shape}, dtype: {s_batch.dtype}")
                print(f"Actions (a_t): {a_batch.shape}, dtype: {a_batch.dtype}")
                print(f"Next States (s_t+1): {s_next_batch.shape}, dtype: {s_next_batch.dtype}")

                # Check action tensor type for discrete vs continuous
                # For Pong (discrete), actions are ints. Our current action_tensor is float.
                # This is fine if the embedding layer handles floats or if they are indices.
                # For nn.Embedding, it expects LongTensor.
                # If using nn.Linear for action embedding, float is okay (e.g. one-hot encoded then to float).
                # Let's adjust ExperienceDataset to handle action dtype more carefully based on env.
            else:
                print("Dataset is empty after collection.")

    except ImportError as e:
        print(f"Import error, likely missing a dependency for the test environment: {e}")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
