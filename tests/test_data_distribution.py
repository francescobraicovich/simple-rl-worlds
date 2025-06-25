"""
Tests for train/validation distribution checking functionality.
"""

import unittest
import numpy as np
import torch
from torch.utils.data import Dataset
import tempfile
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_utils import check_validation_distribution


class MockImageDataset(Dataset):
    """Mock dataset for testing distribution checks"""
    
    def __init__(self, num_samples, mean_shift=0.0, std_scale=1.0, reward_shift=0.0, 
                 image_size=(3, 32, 32), transform=None, config=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.config = config
        
        # Generate synthetic image data
        np.random.seed(42)
        base_images = np.random.rand(num_samples, *image_size).astype(np.float32)
        
        # Apply distribution shifts
        self.states = (base_images + mean_shift) * std_scale
        self.next_states = (np.random.rand(num_samples, *image_size).astype(np.float32) + mean_shift) * std_scale
        
        # Generate actions and rewards
        self.actions = np.random.randint(0, 4, num_samples)  # Discrete actions
        self.rewards = np.random.normal(0.0 + reward_shift, 1.0, num_samples).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        state = torch.from_numpy(self.states[idx])
        next_state = torch.from_numpy(self.next_states[idx])
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        
        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)
        
        return state, action, reward, next_state


class TestDataDistribution(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'environment': {
                'input_channels': 3,
                'image_size': 32
            },
            'data': {
                'save_distribution_plots': False,
                'distribution_plot_dir': tempfile.mkdtemp()
            }
        }
    
    def test_in_distribution_data(self):
        """Test that similar distributions are correctly identified as in-distribution"""
        
        # Create two datasets with similar distributions
        train_dataset = MockImageDataset(100, mean_shift=0.0, std_scale=1.0, reward_shift=0.0, config=self.config)
        val_dataset = MockImageDataset(50, mean_shift=0.05, std_scale=1.02, reward_shift=0.1, config=self.config)
        
        # Check distribution
        result = check_validation_distribution(
            train_dataset, 
            val_dataset, 
            config=self.config,
            n_samples=50,
            save_plots=False
        )
        
        # Should be in distribution (small differences)
        self.assertTrue(result['in_distribution'], 
                       f"Expected in-distribution but got {result['num_significant']}/{result['total_features']} significant features")
        self.assertLess(result['num_significant'], result['total_features'] // 2,
                       "Too many features showing significant differences for similar distributions")
    
    def test_out_of_distribution_data(self):
        """Test that different distributions are correctly identified as out-of-distribution"""
        
        # Create two datasets with very different distributions
        train_dataset = MockImageDataset(100, mean_shift=0.0, std_scale=1.0, reward_shift=0.0, config=self.config)
        val_dataset = MockImageDataset(50, mean_shift=2.0, std_scale=3.0, reward_shift=5.0, config=self.config)
        
        # Check distribution
        result = check_validation_distribution(
            train_dataset, 
            val_dataset, 
            config=self.config,
            n_samples=50,
            save_plots=False
        )
        
        # Should be out of distribution (large differences)
        self.assertFalse(result['in_distribution'], 
                        "Expected out-of-distribution but datasets were classified as in-distribution")
        self.assertGreater(result['num_significant'], 0,
                          "Expected some significant differences but found none")
    
    def test_empty_datasets(self):
        """Test handling of empty datasets"""
        
        empty_dataset = MockImageDataset(0, config=self.config)
        train_dataset = MockImageDataset(10, config=self.config)
        
        # Test empty validation set
        result = check_validation_distribution(
            train_dataset, 
            empty_dataset, 
            config=self.config,
            n_samples=10
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Empty datasets')
    
    def test_feature_extraction(self):
        """Test that feature extraction works correctly"""
        
        train_dataset = MockImageDataset(20, config=self.config)
        val_dataset = MockImageDataset(10, config=self.config)
        
        result = check_validation_distribution(
            train_dataset, 
            val_dataset, 
            config=self.config,
            n_samples=10,
            save_plots=False
        )
        
        # Check that all expected features are present
        expected_features = ['mean_pixel', 'std_pixel', 'median_pixel', 'min_pixel', 
                           'max_pixel', 'q25_pixel', 'q75_pixel', 'reward']
        
        self.assertEqual(len(result['feature_tests']), len(expected_features))
        
        for feature in expected_features:
            self.assertIn(feature, result['feature_tests'])
            feature_result = result['feature_tests'][feature]
            
            # Check that basic statistics are computed
            self.assertIn('train_mean', feature_result)
            self.assertIn('val_mean', feature_result)
            self.assertIn('train_std', feature_result)
            self.assertIn('val_std', feature_result)
    
    def test_plot_generation_disabled(self):
        """Test that plot generation can be disabled"""
        
        train_dataset = MockImageDataset(20, config=self.config)
        val_dataset = MockImageDataset(10, config=self.config)
        
        # Test with plots disabled
        result = check_validation_distribution(
            train_dataset, 
            val_dataset, 
            config=self.config,
            n_samples=10,
            save_plots=False
        )
        
        # Should complete without error
        self.assertIn('dependencies', result)
        self.assertIn('in_distribution', result)
    
    def test_small_sample_handling(self):
        """Test handling of small sample sizes"""
        
        train_dataset = MockImageDataset(5, config=self.config)
        val_dataset = MockImageDataset(3, config=self.config)
        
        result = check_validation_distribution(
            train_dataset, 
            val_dataset, 
            config=self.config,
            n_samples=100,  # Request more samples than available
            save_plots=False
        )
        
        # Should use all available samples
        self.assertEqual(result['train_samples'], 5)
        self.assertEqual(result['val_samples'], 3)
        self.assertIn('in_distribution', result)


if __name__ == '__main__':
    unittest.main()
