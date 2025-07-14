# Contents for src/env_utils.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torchvision import transforms as T

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, k: int):
        """
        Repeats each action k times in the environment.
        Args:
            env: The environment to wrap.
            k (int): Number of times to repeat each action.
        """
        super().__init__(env)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Action repetition factor k must be a positive integer.")
        self.k = k

    def step(self, action):
        """
        Repeats the given action k times, accumulating reward and returning the last observation and info.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None
        current_info = {}
        for _ in range(self.k):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            current_info = info
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, current_info

    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation and info.
        """
        return self.env.reset(**kwargs)


class ImagePreprocessingWrapper(gym.ObservationWrapper):
    """
    Wrapper that applies image preprocessing transformations to observations.
    Outputs uint8 images in (C, H, W) format, range [0,255].
    """
    def __init__(self, env, img_size, grayscale: bool = False):
        """
        Args:
            env: The environment to wrap.
            img_size (tuple): (height, width) for resizing images.
            grayscale (bool): If True, convert images to grayscale.
        """
        super().__init__(env)
        self.img_size = img_size
        self.grayscale = grayscale
        
        # Build PIL-based transforms (no ToTensor, so we stay in uint8)
        transforms = [T.ToPILImage()]
        if self.grayscale:
            transforms.append(T.Grayscale(num_output_channels=1))
        transforms.append(T.Resize(self.img_size))
        transforms.append(T.Lambda(lambda img: T.functional.adjust_contrast(img, 2.0)))
        self.transform = T.Compose(transforms)
        # Determine channels count
        obs_shape = env.observation_space.shape
        channels = 1 if self.grayscale else (obs_shape[2] if len(obs_shape) == 3 else 1)
        # Observation space is now (C, H, W) uint8 [0,255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.img_size[0], self.img_size[1]),
            dtype=np.uint8
        )

    def observation(self, obs):
        """
        Applies preprocessing to the observation.
        Args:
            obs: The observation to process (expects H×W×C or H×W numpy array).
        Returns:
            Processed observation as (C, H, W) uint8 numpy array.
        """
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
    def __init__(self, env, stack_size: int):
        """
        Args:
            env: The environment to wrap.
            stack_size (int): Number of frames to stack.
        """
        super().__init__(env)
        if not isinstance(stack_size, int) or stack_size <= 0:
            raise ValueError("stack_size must be a positive integer.")
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
        """
        Stacks the most recent frames along the channel axis.
        Args:
            obs: Observation array (C, H, W).
        Returns:
            Stacked frames as (C*stack_size, H, W) numpy array.
        """
        if obs.ndim != 3:
            raise ValueError(f"Expected obs of shape (C,H,W), got {obs.shape}")
        c, h, w = obs.shape
        if self.frames is None:
            # initialize by repeating first frame stack_size times
            self.frames = np.repeat(obs[np.newaxis, ...], self.stack_size, axis=0)
            self.frames = self.frames.reshape(c * self.stack_size, h, w)
        else:
            # roll channel blocks: drop oldest block, append new
            stacked = self.frames.reshape(self.stack_size, c, h, w)
            stacked = np.roll(stacked, shift=-1, axis=0)
            stacked[-1] = obs
            self.frames = stacked.reshape(c * self.stack_size, h, w)
        return self.frames.copy()

    def reset(self, **kwargs):
        """
        Resets the environment and frame stack.
        Returns:
            Processed initial observation and info dict.
        """
        self.frames = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

def get_env_details(env_name):

    actual_env_name = env_name

    temp_env = gym.make(actual_env_name)
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
        raise ValueError(
            f"Unsupported action space type: {type(action_space)}")

    temp_env.close()
    return action_dim, action_type, observation_space
