#!/usr/bin/env python3
"""
Test script to verify that the PPO data collection with preprocessing wrappers works correctly.
"""

import sys
sys.path.append('/Users/francescobraicovich/Documents/Progetti/Personali/rl-worlds')

from src.utils.config_utils import load_config
from src.utils.data_utils import collect_ppo_episodes

def test_ppo_collection():
    """Test PPO collection with preprocessing wrappers."""
    
    # Load config
    config_path = '/Users/francescobraicovich/Documents/Progetti/Personali/rl-worlds/config.yaml'
    config = load_config(config_path)
    
    # Update config for quick testing
    config['data']['collection']['num_episodes'] = 2  # Just a few episodes for testing
    config['ppo_agent']['total_train_timesteps'] = 1000  # Quick training
    config['ppo_agent']['n_envs'] = 1  # Single env for simplicity
    config['environment']['name'] = 'CartPole-v1'  # Simple environment
    config['environment']['image_height'] = 32
    config['environment']['image_width'] = 32
    config['environment']['grayscale_conversion'] = True
    config['environment']['frame_stack_size'] = 2
    
    print("Testing PPO collection with preprocessing wrappers...")
    print(f"Environment: {config['environment']['name']}")
    print(f"Image size: {config['environment']['image_height']}x{config['environment']['image_width']}")
    print(f"Grayscale: {config['environment']['grayscale_conversion']}")
    print(f"Frame stack: {config['environment']['frame_stack_size']}")
    
    try:
        # Note: image_size parameter is still needed for interface compatibility
        # but the actual processing is handled by the wrappers
        image_size = (config['environment']['image_height'], config['environment']['image_width'])
        
        train_dataset, val_dataset = collect_ppo_episodes(
            config=config,
            max_steps_per_episode=100,  # Short episodes for testing
            image_size=image_size,
            validation_split_ratio=0.2,
            frame_skipping=0
        )
        
        print("\\nSuccess! Collected data:")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        if len(train_dataset) > 0:
            sample_state, sample_action, sample_reward, sample_next_state = train_dataset[0]
            print("\\nSample data shapes:")
            print(f"State shape: {sample_state.shape}")
            print(f"Action: {sample_action}")
            print(f"Reward: {sample_reward}")
            print(f"Next state shape: {sample_next_state.shape}")
            
            # Verify preprocessing worked correctly
            expected_channels = config['environment']['frame_stack_size']  # Should be 2 frames * 1 channel (grayscale)
            if sample_state.shape[0] == expected_channels:
                print(f"✓ Frame stacking working correctly: {expected_channels} channels")
            else:
                print(f"✗ Frame stacking issue: expected {expected_channels}, got {sample_state.shape[0]}")
                
            if sample_state.shape[1:] == (32, 32):
                print(f"✓ Image resizing working correctly: {sample_state.shape[1:]} shape")
            else:
                print(f"✗ Image resizing issue: expected (32, 32), got {sample_state.shape[1:]}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ppo_collection()
    if success:
        print("\\n🎉 Test completed successfully!")
    else:
        print("\\n❌ Test failed!")
        sys.exit(1)
