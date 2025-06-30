#!/usr/bin/env python3
"""
Test script to verify torch.save/torch.load works with ExperienceDataset
"""

import torch
import os
import tempfile
from src.utils.data_utils import ExperienceDataset

def test_torch_save_load():
    """Test that ExperienceDataset can be saved and loaded with torch.save/torch.load"""
    
    # Create sample data (with proper stacking)
    states = [torch.randn(1, 3, 64, 64) for _ in range(10)]  # 1 frame stack, 3 channels
    actions = [torch.tensor([0.1, 0.2]) for _ in range(10)]
    rewards = [torch.tensor(1.0) for _ in range(10)]
    next_states = [torch.randn(1, 3, 64, 64) for _ in range(10)]  # 1 frame stack, 3 channels
    
    # Create sample config
    config = {
        'environment': {
            'image_height': 64,
            'image_width': 64,
            'frame_stack_size': 1,
            'input_channels_per_frame': 3,
            'grayscale_conversion': False
        }
    }
    
    # Create dataset
    original_dataset = ExperienceDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        transform=None,  # Important: transform should be None for saved datasets
        config=config
    )
    
    # Test data structure similar to what's saved in the main code
    data_to_save = {
        'train_dataset': original_dataset,
        'val_dataset': original_dataset,  # Using same for simplicity
        'metadata': {
            'environment_name': 'test_env',
            'num_episodes_collected': 5,
            'image_height': 64,
            'image_width': 64,
            'collection_method': 'test'
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Save with torch.save
        print("Saving dataset with torch.save...")
        torch.save(data_to_save, temp_path)
        print("Save successful!")
        
        # Load with torch.load
        print("Loading dataset with torch.load...")
        loaded_data = torch.load(temp_path, weights_only=False)
        print("Load successful!")
        
        # Verify structure
        assert 'train_dataset' in loaded_data
        assert 'val_dataset' in loaded_data
        assert 'metadata' in loaded_data
        
        # Verify dataset properties
        loaded_train_dataset = loaded_data['train_dataset']
        assert len(loaded_train_dataset) == len(original_dataset)
        
        # Test a sample from the loaded dataset
        state, action, reward, next_state = loaded_train_dataset[0]
        print(f"Sample data shapes - State: {state.shape}, Action: {action.shape}, Reward: {reward.shape}, Next State: {next_state.shape}")
        
        # Verify metadata
        metadata = loaded_data['metadata']
        assert metadata['environment_name'] == 'test_env'
        assert metadata['collection_method'] == 'test'
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    test_torch_save_load()
