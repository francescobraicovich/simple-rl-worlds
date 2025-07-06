import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torchvision import transforms as T

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        if k <= 0:
            raise ValueError("Action repetition factor k must be positive.")
        self.k = k
        self.env = env # Redundant, super().__init__(env) does this. Kept for clarity if needed.

        # The observation space and action space remain the same as the wrapped environment.
        # self.observation_space = env.observation_space
        # self.action_space = env.action_space

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        current_info = {} # To store info, potentially from the last step or aggregated

        for _ in range(self.k):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            current_info = info # Keep the info from the last step
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, current_info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ImagePreprocessingWrapper(gym.ObservationWrapper):
    """
    Wrapper that applies image preprocessing transformations to observations.
    This allows PPO to train on the preprocessed images directly.
    """
    def __init__(self, env, image_size, grayscale=False):
        super().__init__(env)
        self.image_size = image_size  # (height, width)
        self.grayscale = grayscale
        
        # Create the preprocessing transforms
        # Note: T.ToTensor() normalizes to [0, 1], so we'll handle conversion manually
        transforms = [T.ToPILImage()]
        if self.grayscale:
            transforms.append(T.Grayscale(num_output_channels=1))
        transforms.append(T.Resize(self.image_size))
        self.transform = T.Compose(transforms)
        
        # Update observation space to (height, width, channels) format for SB3 compatibility
        # Use uint8 [0, 255] for SB3 image space recognition, even though we return float32 [0, 1]
        if self.grayscale:
            channels = 1
        else:
            channels = env.observation_space.shape[2] if len(env.observation_space.shape) == 3 else 1
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size[0], self.image_size[1], channels),  # (H, W, C) format
            dtype=np.uint8
        )

    def observation(self, obs):
        # Handle different observation types
        if isinstance(obs, np.ndarray):
            if obs.dtype != np.uint8:
                # Convert to uint8 if not already
                obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)
        else:
            # Try to render if observation is not an image
            if hasattr(self.env, 'render') and callable(self.env.render):
                try:
                    obs = self.env.render()
                    if obs is None or not isinstance(obs, np.ndarray):
                        raise ValueError("Render returned invalid observation")
                except Exception:
                    raise ValueError(f"Cannot process observation type: {type(obs)}")
            else:
                raise ValueError(f"Cannot process observation type: {type(obs)}")
        
        # Ensure the observation has the right shape
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=2)  # Add channel dimension
        elif len(obs.shape) != 3:
            raise ValueError(f"Observation must be 2D or 3D, got shape: {obs.shape}")
        
        # Apply transforms (PIL operations) and convert back to numpy
        # Note: We don't use T.ToTensor() to avoid normalization
        transformed_pil = self.transform(obs)
        
        # Convert PIL back to numpy array, maintaining uint8 [0, 255] range
        transformed_np = np.array(transformed_pil, dtype=np.uint8)
        
        # Ensure we have the right dimensions (H, W, C)
        if len(transformed_np.shape) == 2:
            # Grayscale image, add channel dimension
            transformed_np = np.expand_dims(transformed_np, axis=2)
        elif len(transformed_np.shape) == 3:
            # Already has channels, keep as is
            pass
        else:
            raise ValueError(f"Unexpected transformed image shape: {transformed_np.shape}")
        
        return transformed_np


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Wrapper that stacks the last N frames together.
    Expects input in (H, W, C) format and outputs in (H, W, C*stack_size) format.
    """
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = None
        
        # Update observation space
        old_space = env.observation_space
        if len(old_space.shape) == 3:
            # Multiply channels by stack size: (H, W, C) -> (H, W, C*stack_size)
            new_shape = (old_space.shape[0], old_space.shape[1], old_space.shape[2] * stack_size)
        else:
            raise ValueError(f"Expected 3D observation space, got: {old_space.shape}")
        
        self.observation_space = spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype
        )

    def observation(self, obs):
        if self.frames is None:
            # Initialize with the first observation repeated
            self.frames = np.concatenate([obs] * self.stack_size, axis=2)  # Stack along channel dimension
        else:
            # Shift frames and add new observation
            channels_per_frame = obs.shape[2]
            self.frames = np.concatenate([
                self.frames[:, :, channels_per_frame:],  # Remove oldest frame
                obs  # Add newest frame
            ], axis=2)
        
        return self.frames.copy()

    def reset(self, **kwargs):
        self.frames = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

if __name__ == '__main__':
    # Example Usage (requires a Gymnasium environment to be installed, e.g., pip install gymnasium[classic_control])
    try:
        env = gym.make("CartPole-v1")
        print(f"Original env: {env}")

        k_repeat = 4
        wrapped_env = ActionRepeatWrapper(env, k_repeat)
        print(f"Wrapped env with k={k_repeat}: {wrapped_env}")

        obs, info = wrapped_env.reset(seed=42)
        print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

        action = wrapped_env.action_space.sample()
        print(f"Taking action: {action}")

        next_obs, total_reward, terminated, truncated, info = wrapped_env.step(action)

        print(f"Next observation shape: {next_obs.shape if hasattr(next_obs, 'shape') else 'N/A'}")
        print(f"Total reward after {k_repeat} steps: {total_reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"Info: {info}")

        # Test episode completion
        done = False
        episode_reward = 0
        steps = 0
        obs, info = wrapped_env.reset(seed=123)
        while not done:
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            episode_reward += reward
            steps +=1
            done = terminated or truncated
            if done:
                print(f"Episode finished after {steps} wrapped steps. Total reward: {episode_reward}")
                break

        wrapped_env.close()
        env.close() # Original env also closed by wrapped_env.close if it's the one it holds.

    except ImportError:
        print("Gymnasium or specific environment not installed. Skipping example usage.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
