import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
