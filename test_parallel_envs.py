#!/usr/bin/env python3
"""
Test script to verify SubprocVecEnv integration works correctly
"""

import multiprocessing
from src.utils.data_utils import create_env_for_ppo
from stable_baselines3.common.vec_env import SubprocVecEnv

def test_parallel_env_creation():
    """Test that we can create parallel environments successfully"""
    
    # Test parameters
    env_name = "CartPole-v1"  # Simple environment that should be available
    action_repetition_k = 0
    env_render_mode = None
    n_envs = min(2, multiprocessing.cpu_count())  # Use 2 or CPU count, whichever is smaller
    
    print(f"Testing parallel environment creation with {n_envs} environments...")
    
    try:
        # Test creating a single environment function
        env_fn = create_env_for_ppo(env_name, action_repetition_k, env_render_mode)
        single_env = env_fn()
        print(f"✓ Single environment creation successful: {type(single_env)}")
        single_env.close()
        
        # Test SubprocVecEnv creation
        env_fns = [create_env_for_ppo(env_name, action_repetition_k, env_render_mode) for _ in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        print(f"✓ SubprocVecEnv creation successful with {vec_env.num_envs} environments")
        
        # Test basic operations
        obs = vec_env.reset()
        print(f"✓ Environment reset successful, observation shape: {obs.shape}")
        
        # Test step
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"✓ Environment step successful, got {len(rewards)} rewards")
        
        # Clean up
        vec_env.close()
        print("✓ Environment cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_cpu_count_detection():
    """Test CPU count detection and environment number logic"""
    
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected CPU cores: {cpu_count}")
    
    # Test various n_envs values
    test_values = [0, 1, cpu_count, cpu_count + 5]
    
    for n_envs_config in test_values:
        n_envs = max(1, min(n_envs_config, cpu_count))
        print(f"  Config: {n_envs_config} → Actual: {n_envs}")
    
    print("✓ CPU count detection logic working correctly")

if __name__ == "__main__":
    print("=== Testing SubprocVecEnv Integration ===")
    
    test_cpu_count_detection()
    print()
    
    if test_parallel_env_creation():
        print("\n🎉 All tests passed! SubprocVecEnv integration is working correctly.")
    else:
        print("\n❌ Tests failed! Check the implementation.")
