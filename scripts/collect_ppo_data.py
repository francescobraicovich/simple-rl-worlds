# This script will be used for PPO agent training and data collection.
# It will incorporate action repetition and frame stacking.

import argparse
import os
import pickle
import time
import torch
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from src.utils.config_utils import load_config
# from src.utils.env_utils import get_env_details # May not be strictly needed
from src.utils.env_wrappers import ActionRepeatWrapper # Corrected import path
from src.rl_agent import create_ppo_agent, train_ppo_agent

def collect_data(args):
    """
    Trains a PPO agent and collects transition data (s, a, r, s', done).
    Action repetition and frame stacking are applied based on config.
    Single frames are saved from stacked observations.
    """
    config = load_config(args.config)

    # Device Setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Parameters from Config
    env_name = config['environment_name']
    ppo_config = config['ppo_agent']

    # Action repetition factor 'k' from ppo_agent config, frame_stack_k from main config
    action_repetition_k = ppo_config.get('action_repetition_k', 1) # For repeating actions

    # Frame stacking amount - assuming this is what input_channels refers to in terms of final stacked shape
    # For VecFrameStack, n_stack is the number of frames to stack.
    # If the original environment has C channels, and we stack N frames,
    # the stacked observation will have C*N channels.
    # The config 'input_channels' seems to refer to the *original* channels of a single frame.
    # Let's assume VecFrameStack's n_stack should be the same as action_repetition_k for simplicity,
    # as per the problem's context of using 'k' for both.
    # However, it's more robust to have a separate config for n_stack if they can differ.
    # For now, let's use action_repetition_k as the n_stack value, which seems implied.
    frame_stack_n = action_repetition_k # Number of frames to stack in VecFrameStack

    num_episodes_collect = config.get('num_episodes_data_collection', 50)
    max_steps_per_episode_collect = config.get('max_steps_per_episode_data_collection', 450)
    dataset_dir = config.get('dataset_dir', 'datasets/')
    dataset_filename = config.get('dataset_filename', 'collected_data.pkl')

    # This is the number of channels of a *single* frame from the original environment
    original_input_channels = config.get('input_channels', 3)

    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Action repetition factor k: {action_repetition_k}")
    print(f"Frame stack n: {frame_stack_n}")

    # Environment Setup Function
    def make_env_fn():
        env = gym.make(env_name)
        env = Monitor(env) # Monitor should wrap the base env for episode stats
        if action_repetition_k > 1:
            env = ActionRepeatWrapper(env, action_repetition_k)
        return env

    vec_env = DummyVecEnv([make_env_fn])

    if frame_stack_n > 1:
        print(f"Wrapping environment with VecFrameStack, n_stack={frame_stack_n}")
        # Default channels_order='stable-baselines3' means (C, H, W) and stacks to (num_channels * n_stack, H, W)
        # If the base env is (H, W, C), SB3 wrappers usually handle this.
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack_n, channels_order='stable-baselines3')

    print(f"Observation space after potential stacking: {vec_env.observation_space.shape}")
    print(f"Action space: {vec_env.action_space}")

    # PPO Agent
    print("Creating PPO agent...")
    ppo_agent = create_ppo_agent(vec_env, ppo_config, device=device)
    print("Training PPO agent...")
    train_ppo_agent(ppo_agent, ppo_config, task_name="PPO Training for Data Collection")

    # Data Collection
    print(f"\nStarting data collection for {num_episodes_collect} episodes...")
    collected_transitions = []

    # Reset returns an observation for each environment in the vector.
    # For DummyVecEnv with one env, obs shape is (1, C_stacked, H, W)
    obs = vec_env.reset()

    episodes_completed = 0
    current_episode_steps = 0
    start_time = time.time()

    while episodes_completed < num_episodes_collect:
        action, _ = ppo_agent.predict(obs, deterministic=False) # Use stochastic actions for exploration
        next_obs, rewards, dones, infos = vec_env.step(action)

        # Extract the most recent single frame from the stacked observation
        # obs and next_obs from VecFrameStack are (num_envs, C_stacked, H, W)
        # C_stacked = original_input_channels * frame_stack_n
        # We want to save the latest single frame: (original_input_channels, H, W)

        # Assuming channels_order='stable-baselines3' which is (C,H,W)
        # The last 'original_input_channels' in the channel dimension correspond to the most recent frame
        if frame_stack_n > 1:
            single_obs_to_save = obs[0, -original_input_channels:, :, :]
            single_next_obs_to_save = next_obs[0, -original_input_channels:, :, :]
        else: # If no frame stacking, obs is already the single frame (potentially with a channel dim)
              # obs shape is (1, original_input_channels, H, W)
            single_obs_to_save = obs[0]
            single_next_obs_to_save = next_obs[0]

        collected_transitions.append((
            single_obs_to_save,
            action[0],          # Action is a numpy array for DummyVecEnv
            rewards[0],         # Reward is a numpy array
            single_next_obs_to_save,
            bool(dones[0])      # Done is a numpy array of booleans
        ))

        obs = next_obs
        current_episode_steps += 1

        if dones[0]: # done from Monitor refers to a true episode end (terminated or truncated by TimeLimit)
            episode_info = infos[0].get("episode")
            if episode_info:
                print(f"Episode {episodes_completed + 1}/{num_episodes_collect} finished. "
                      f"Reward: {episode_info['r']:.2f}, Length: {episode_info['l']}. "
                      f"Total transitions: {len(collected_transitions)}")
            else: # Should not happen with Monitor
                print(f"Episode {episodes_completed + 1}/{num_episodes_collect} finished (info not found). "
                      f"Total transitions: {len(collected_transitions)}")

            episodes_completed += 1
            current_episode_steps = 0
            if episodes_completed < num_episodes_collect:
                # obs = vec_env.reset() # Reset is handled by DummyVecEnv's auto-reset on done
                pass # DummyVecEnv automatically resets, obs is already the new state
        elif current_episode_steps >= max_steps_per_episode_collect:
            print(f"Episode {episodes_completed + 1}/{num_episodes_collect} reached max steps ({max_steps_per_episode_collect}). Resetting. "
                  f"Total transitions: {len(collected_transitions)}")
            # This case is mostly for environments without internal time limits or if Monitor is not used,
            # but Monitor's TimeLimit wrapper should trigger dones[0]=True before this.
            # Force reset if max_steps_per_episode_collect is reached without a natural done.
            obs = vec_env.reset() # Manually reset if max steps hit (safety)
            episodes_completed += 1 # Count as a completed episode for data collection purposes
            current_episode_steps = 0


    end_time = time.time()
    print(f"Data collection finished. Time taken: {end_time - start_time:.2f} seconds.")

    # Save Data
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    try:
        with open(dataset_path, 'wb') as f:
            pickle.dump(collected_transitions, f)
        print(f"Collected {len(collected_transitions)} transitions and saved to {dataset_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

    vec_env.close()
    print("Environment closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect data using a PPO agent with action repetition and frame stacking.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')

    args = parser.parse_args()

    # Ensure the script is run from the root directory for correct relative paths
    if not os.path.exists(args.config) and not args.config.startswith('/'):
        # Basic check, assumes script might be in 'scripts/' and config at root
        print(f"Warning: Config file '{args.config}' not found in current directory.")
        potential_root_config_path = os.path.join("..", args.config)
        if os.path.exists(potential_root_config_path):
            print(f"Found '{potential_root_config_path}', adjusting path.")
            args.config = potential_root_config_path
        else:
            print(f"Config file '{args.config}' not found. Please ensure paths are correct.")
            # exit(1) # Consider exiting if config is critical and not found

    collect_data(args)
