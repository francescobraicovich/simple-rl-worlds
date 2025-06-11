# Contents for src/env_utils.py
import gymnasium as gym
import random
import ale_py

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
    print(f"Environment: {actual_env_name}")
    print(f"Action space type: {action_type}, Action dimension: {action_dim}")
    print("Raw observation space:", observation_space)
    return action_dim, action_type, observation_space
