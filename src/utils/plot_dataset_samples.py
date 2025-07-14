#!/usr/bin/env python3
"""
Dataset Sample Visualization Utility

This script provides functionality for visualizing samples from a training dataset.
It loads configuration settings, loads a training dataset using the collect_load_data pipeline,
samples a number of (state, action, reward, next_state) tuples, and generates plots showing
current state frames and corresponding next state frames with actions and rewards.

Key Features:
- Loads configuration from config.yaml
- Uses the DataLoadingPipeline to load existing datasets
- Samples and visualizes state transitions
- Handles grayscale image data with proper channel dimensions
- Saves visualizations to a subdirectory in the dataset folder

Usage:
    python src/utils/plot_dataset_samples.py [--config CONFIG_PATH] [--num_samples NUM] [--output_dir OUTPUT]
"""

import sys
import os
import argparse
import random
from pathlib import Path
from typing import Tuple, List

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add project root to path for imports (two levels up from utils)
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_utils import load_config
from src.scripts.collect_load_data import DataLoadingPipeline


def setup_matplotlib():
    """Configure matplotlib for better plots."""
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8


def process_state_tensor(state_tensor: torch.Tensor) -> np.ndarray:
    """
    Process state tensor for visualization.
    
    Args:
        state_tensor: State tensor with shape [1, T, H, W] or [T, H, W]
        
    Returns:
        Numpy array suitable for matplotlib visualization
    """
    # Remove batch dimension if present
    if state_tensor.dim() == 4 and state_tensor.shape[0] == 1:
        state_tensor = state_tensor.squeeze(0)  # [T, H, W]
    
    # Convert to numpy and ensure proper data type
    state_np = state_tensor.detach().cpu().numpy()
    
    # Normalize to [0, 1] range if needed
    if state_np.max() > 1.0:
        state_np = state_np / 255.0
    
    return state_np


def process_next_state_tensor(next_state_tensor: torch.Tensor) -> np.ndarray:
    """
    Process next state tensor for visualization.
    
    Args:
        next_state_tensor: Next state tensor with shape [1, 1, H, W] or [1, H, W]
        
    Returns:
        Numpy array suitable for matplotlib visualization
    """
    # Remove batch dimension if present
    if next_state_tensor.dim() == 4 and next_state_tensor.shape[0] == 1:
        next_state_tensor = next_state_tensor.squeeze(0)  # [1, H, W]
    
    # Remove temporal dimension (should be 1)
    if next_state_tensor.dim() == 3 and next_state_tensor.shape[0] == 1:
        next_state_tensor = next_state_tensor.squeeze(0)  # [H, W]
    
    # Convert to numpy and ensure proper data type
    next_state_np = next_state_tensor.detach().cpu().numpy()
    
    # Normalize to [0, 1] range if needed
    if next_state_np.max() > 1.0:
        next_state_np = next_state_np / 255.0
    
    return next_state_np


def create_sample_visualization(sample_data: List[Tuple], config: dict, output_path: str):
    """
    Create a comprehensive visualization of dataset samples.
    
    Args:
        sample_data: List of (state, next_state, action, reward) tuples
        config: Configuration dictionary
        output_path: Path to save the visualization
    """
    num_samples = len(sample_data)
    sequence_length = config['data_and_patching']['sequence_length']
    
    # Calculate grid dimensions
    cols = min(4, num_samples)  # Max 4 columns
    rows = (num_samples + cols - 1) // cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5 * rows))
    
    for i, (state, next_state, action, reward) in enumerate(sample_data):
        # Process tensors
        state_np = process_state_tensor(state)  # [T, H, W]
        next_state_np = process_next_state_tensor(next_state)  # [H, W]
        
        # Create subplot for this sample
        row = i // cols
        col = i % cols
        
        # Create a gridspec for this sample (state frames + next state)
        sample_gs = gridspec.GridSpec(2, sequence_length + 1, 
                                    figure=fig,
                                    left=col/cols, right=(col+1)/cols,
                                    top=1-row/(rows), bottom=1-(row+1)/(rows),
                                    hspace=0.3, wspace=0.1)
        
        # Plot state frames (sequence)
        for t in range(sequence_length):
            ax = fig.add_subplot(sample_gs[0, t])
            ax.imshow(state_np[t], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'State t-{sequence_length-1-t}', fontsize=8)
            ax.axis('off')
        
        # Plot next state
        ax = fig.add_subplot(sample_gs[0, sequence_length])
        ax.imshow(next_state_np, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Next State', fontsize=8, color='red')
        ax.axis('off')
        
        # Plot action and reward information
        ax_info = fig.add_subplot(sample_gs[1, :])
        ax_info.axis('off')
        
        # Format action and reward information
        if action.dim() > 0:
            action_str = f"Actions: {action.tolist()}"
        else:
            action_str = f"Action: {action.item()}"
            
        if reward.dim() > 0:
            reward_str = f"Rewards: {[f'{r:.3f}' for r in reward.tolist()]}"
        else:
            reward_str = f"Reward: {reward.item():.3f}"
        
        info_text = f"Sample {i+1}\n{action_str}\n{reward_str}"
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Dataset Sample Visualization - {num_samples} Samples\n'
                f'Environment: {config["environment"]["name"]} | '
                f'Sequence Length: {sequence_length}', 
                fontsize=14, y=0.98)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved sample visualization to: {output_path}")


def create_individual_sample_plots(sample_data: List[Tuple], config: dict, output_dir: str):
    """
    Create individual plots for each sample.
    
    Args:
        sample_data: List of (state, next_state, action, reward) tuples
        config: Configuration dictionary
        output_dir: Directory to save individual plots
    """
    sequence_length = config['data_and_patching']['sequence_length']
    
    for i, (state, next_state, action, reward) in enumerate(sample_data):
        # Process tensors
        state_np = process_state_tensor(state)  # [T, H, W]
        next_state_np = process_next_state_tensor(next_state)  # [H, W]
        
        # Create figure for this sample
        fig, axes = plt.subplots(2, sequence_length + 1, figsize=(15, 6))
        
        # Plot state sequence
        for t in range(sequence_length):
            ax = axes[0, t]
            ax.imshow(state_np[t], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'State t-{sequence_length-1-t}')
            ax.axis('off')
        
        # Plot next state
        ax = axes[0, sequence_length]
        ax.imshow(next_state_np, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Next State', color='red')
        ax.axis('off')
        
        # Clear bottom row and add info
        for ax in axes[1, :]:
            ax.axis('off')
        
        # Action and reward info
        if action.dim() > 0:
            action_text = f"Actions: {action.tolist()}"
        else:
            action_text = f"Action: {action.item()}"
            
        if reward.dim() > 0:
            reward_text = f"Rewards: {[f'{r:.3f}' for r in reward.tolist()]}"
        else:
            reward_text = f"Reward: {reward.item():.3f}"
        
        axes[1, sequence_length//2].text(0.5, 0.5, f"{action_text}\n{reward_text}", 
                                       ha='center', va='center', fontsize=12,
                                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'Sample {i+1} - State Transition Visualization', fontsize=14)
        
        output_path = os.path.join(output_dir, f'sample_{i+1:03d}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    print(f"✓ Saved {len(sample_data)} individual sample plots to: {output_dir}")


def sample_dataset(dataset, num_samples: int) -> List[Tuple]:
    """
    Sample random data points from the dataset.
    
    Args:
        dataset: The ExperienceDataset to sample from
        num_samples: Number of samples to extract
        
    Returns:
        List of (state, next_state, action, reward) tuples
    """
    if num_samples > len(dataset):
        print(f"Warning: Requested {num_samples} samples but dataset only has {len(dataset)}. Using all samples.")
        num_samples = len(dataset)
    
    # Sample random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    sample_data = []
    for idx in indices:
        state, next_state, action, reward = dataset[idx]
        sample_data.append((state, next_state, action, reward))
    
    return sample_data


def load_dataset_pipeline(config_path: str):
    """
    Load dataset using the DataLoadingPipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, config)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize data loading pipeline
    batch_size = 1  # Use batch size of 1 for sampling
    pipeline = DataLoadingPipeline(batch_size=batch_size, config_path=config_path)
    
    # Load data
    print("Loading dataset using DataLoadingPipeline...")
    train_dataloader, val_dataloader = pipeline.run_pipeline()
    
    # Get the underlying dataset
    train_dataset = pipeline.train_dataset
    val_dataset = pipeline.val_dataset
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset) if val_dataset else 0}")
    
    return train_dataset, val_dataset, config


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Visualize samples from RL training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to visualize"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: datasets/visualizations/)"
    )
    
    parser.add_argument(
        "--individual_plots",
        action="store_false",
        help="Disable individual plots (by default, only individual plots are created)"
    )
    
    parser.add_argument(
        "--combined_plot",
        action="store_true",
        help="Create combined visualization in addition to individual plots"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup matplotlib
    setup_matplotlib()
    
    try:
        # Load dataset
        train_dataset, val_dataset, config = load_dataset_pipeline(args.config)
        
        # Use training dataset for visualization
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        
        # Sample data points
        print(f"Sampling {args.num_samples} data points from training dataset...")
        sample_data = sample_dataset(train_dataset, args.num_samples)
        
        # Setup output directory
        if args.output_dir is None:
            output_dir = os.path.join("datasets", "visualizations")
        else:
            output_dir = args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create individual plots by default
        if args.individual_plots:
            individual_dir = os.path.join(output_dir, "individual_samples")
            os.makedirs(individual_dir, exist_ok=True)
            create_individual_sample_plots(sample_data, config, individual_dir)
        
        # Create combined visualization if requested
        if args.combined_plot:
            combined_output_path = os.path.join(output_dir, "dataset_samples_combined.png")
            create_sample_visualization(sample_data, config, combined_output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATASET VISUALIZATION SUMMARY")
        print("=" * 60)
        print(f"Environment: {config['environment']['name']}")
        print(f"Sequence Length: {config['data_and_patching']['sequence_length']}")
        print(f"Image Size: {config['data_and_patching']['image_height']}x{config['data_and_patching']['image_width']}")
        print(f"Samples Visualized: {len(sample_data)}")
        print(f"Output Directory: {output_dir}")
        if args.individual_plots:
            individual_dir = os.path.join(output_dir, "individual_samples")
            print(f"Individual Plots: {individual_dir}")
        if args.combined_plot:
            combined_output_path = os.path.join(output_dir, "dataset_samples_combined.png")
            print(f"Combined Plot: {combined_output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
