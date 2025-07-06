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
    Outputs uint8 images in (C, H, W) format, range [0,255].
    """
    def __init__(self, env, image_size, grayscale=False):
        super().__init__(env)
        self.image_size = image_size  # (height, width)
        self.grayscale = grayscale

        # Build PIL-based transforms (no ToTensor, so we stay in uint8)
        transforms = [T.ToPILImage()]
        if self.grayscale:
            transforms.append(T.Grayscale(num_output_channels=1))
        transforms.append(T.Resize(self.image_size))
        self.transform = T.Compose(transforms)

        # Determine channels count
        channels = 1 if self.grayscale else (
            env.observation_space.shape[2] if len(env.observation_space.shape) == 3 else 1
        )

        # Observation space is now (C, H, W) uint8 [0,255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.image_size[0], self.image_size[1]),
            dtype=np.uint8
        )

    def observation(self, obs):
        # 1) Get a uint8 H×W×C numpy array
        if not isinstance(obs, np.ndarray):
            obs = self.env.render()
            if not isinstance(obs, np.ndarray):
                raise ValueError(f"Cannot process observation type: {type(obs)}")
        if obs.dtype != np.uint8:
            # assume in [0,1] floats or other ints
            obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)

        # 2) Ensure it’s H×W×C
        if obs.ndim == 2:
            obs = obs[:, :, None]
        elif obs.ndim != 3:
            raise ValueError(f"Expected 2D or 3D obs, got shape: {obs.shape}")

        # 3) PIL transforms & back to uint8 array (H,W) or (H,W,C)
        pil = self.transform(obs)
        arr = np.array(pil, dtype=np.uint8)

        # 4) If grayscale, ensure channel dimension
        if arr.ndim == 2:
            arr = arr[:, :, None]

        # 5) Convert to (C, H, W)
        arr = arr.transpose(2, 0, 1)

        return arr

class FrameStackWrapper(gym.ObservationWrapper):
    """
    Wrapper that stacks the last N frames together.
    Expects and returns observations in (C, H, W) format,
    stacking along the channel axis to produce (C*stack_size, H, W).
    """
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = None

        old_space = env.observation_space
        if len(old_space.shape) != 3:
            raise ValueError(f"Expected 3D observation space, got: {old_space.shape}")
        c, h, w = old_space.shape

        # New shape: (C * stack_size, H, W)
        new_shape = (c * stack_size, h, w)
        self.observation_space = spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype
        )

    def observation(self, obs):
        # obs: numpy array (C, H, W)
        if obs.ndim != 3:
            raise ValueError(f"Expected obs of shape (C,H,W), got {obs.shape}")
        c, h, w = obs.shape

        if self.frames is None:
            # initialize by repeating first frame stack_size times
            self.frames = np.repeat(obs[np.newaxis, ...], self.stack_size, axis=0)
            # reshape to (C*stack_size, H, W)
            self.frames = self.frames.reshape(c * self.stack_size, h, w)
        else:
            # roll channel blocks: drop oldest block, append new
            # reshape to (stack_size, C, H, W)
            stacked = self.frames.reshape(self.stack_size, c, h, w)
            stacked = np.roll(stacked, shift=-1, axis=0)
            stacked[-1] = obs  # insert newest
            self.frames = stacked.reshape(c * self.stack_size, h, w)

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
