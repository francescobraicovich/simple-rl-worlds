import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import random
import re

VMAX = 255

def cleanup_future_epoch_images(output_dir: Path, current_epoch: int, model_name: str) -> None:
    """
    Remove images from future epochs (higher epoch numbers than current).
    
    Args:
        output_dir: Directory containing the images
        current_epoch: Current epoch number
        model_name: Model name to match in filenames
    """
    if not output_dir.exists():
        return
    
    # Pattern to match epoch numbers in filenames: model_name_epoch_XXX_*.png
    pattern = re.compile(rf"{model_name}_epoch_(\d+)_.*\.png")
    
    files_deleted = 0
    for file_path in output_dir.glob("*.png"):
        match = pattern.match(file_path.name)
        if match:
            file_epoch = int(match.group(1))
            if file_epoch > current_epoch:
                try:
                    file_path.unlink()  # Delete the file
                    files_deleted += 1
                except OSError as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
    
    if files_deleted > 0:
        print(f"Deleted {files_deleted} image(s) from future epochs (>{current_epoch})")


def plot_validation_samples(
    true_next_states: torch.Tensor,
    predicted_next_states: torch.Tensor,
    epoch: int,
    sample_indices: List[int],
    output_dir: str,
    model_name: str = "encoder_decoder"
) -> None:
    """
    Plot true vs predicted next states for validation samples.
    
    Args:
        true_next_states: Ground truth next states [B, T, H, W]
        predicted_next_states: Predicted next states [B, T, H, W] 
        epoch: Current epoch number
        sample_indices: List of sample indices to plot (should be 5 samples)
        output_dir: Directory to save plots (e.g., "evaluation_plots/decoder_plots/encoder_decoder")
        model_name: Name of the model for plot title
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up any images from future epochs
    cleanup_future_epoch_images(output_path, epoch, model_name)
    
    # Convert tensors to numpy and move to CPU
    true_states = true_next_states.detach().cpu().numpy()
    pred_states = predicted_next_states.detach().cpu().numpy()
    
    # Convert from [0, 1] range back to [0, 255] for plotting
    true_states = (true_states * 255).astype('uint8')
    pred_states = (pred_states * 255).astype('uint8')
    
    # Get dimensions
    batch_size, T, H, W = true_states.shape
    
    # Select the 5 samples using provided indices
    n_samples = min(5, len(sample_indices), batch_size)
    selected_indices = sample_indices[:n_samples]
    
    # Create figure with 2 rows (true vs predicted) and T columns for time frames
    fig, axes = plt.subplots(2, T, figsize=(3*T, 6))
    
    # If T=1, axes might not be 2D, so ensure it is
    if T == 1:
        axes = axes.reshape(2, 1)
    
    # Plot each sample
    for sample_idx, batch_idx in enumerate(selected_indices):
        if sample_idx >= n_samples:
            break
            
        # Create a separate figure for each sample
        fig_sample, axes_sample = plt.subplots(2, T, figsize=(3*T, 6))
        if T == 1:
            axes_sample = axes_sample.reshape(2, 1)
        
        # Plot true frames (top row)
        for t in range(T):
            axes_sample[0, t].imshow(true_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes_sample[0, t].set_title(f'True Frame {t+1}')
            axes_sample[0, t].axis('off')
        
        # Plot predicted frames (bottom row)
        for t in range(T):
            axes_sample[1, t].imshow(pred_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes_sample[1, t].set_title(f'Predicted Frame {t+1}')
            axes_sample[1, t].axis('off')
        
        # Add overall title
        fig_sample.suptitle(f'{model_name.replace("_", " ").title()} - Epoch {epoch} - Sample {sample_idx+1}', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        filename = f"{model_name}_epoch_{epoch:03d}_sample_{sample_idx+1}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_sample)
    
    print(f"Saved validation plots for epoch {epoch} to {output_path}")


def plot_combined_validation_samples(
    true_next_states: torch.Tensor,
    predicted_next_states: torch.Tensor,
    epoch: int,
    sample_indices: List[int],
    output_dir: str,
    model_name: str = "encoder_decoder"
) -> None:
    """
    Plot all 5 samples in a single combined figure.
    
    Args:
        true_next_states: Ground truth next states [B, T, H, W]
        predicted_next_states: Predicted next states [B, T, H, W] 
        epoch: Current epoch number
        sample_indices: List of sample indices to plot (should be 5 samples)
        output_dir: Directory to save plots
        model_name: Name of the model for plot title
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up any images from future epochs
    cleanup_future_epoch_images(output_path, epoch, model_name)
    
    # Convert tensors to numpy and move to CPU
    true_states = true_next_states.detach().cpu().numpy()
    pred_states = predicted_next_states.detach().cpu().numpy()
    
    # Convert from [0, 1] range back to [0, 255] for plotting
    true_states = (true_states * 255).astype('uint8')
    pred_states = (pred_states * 255).astype('uint8')
    
    # Get dimensions
    batch_size, T, H, W = true_states.shape
    
    # Select the 5 samples using provided indices
    n_samples = min(5, len(sample_indices), batch_size)
    selected_indices = sample_indices[:n_samples]
    
    # Create figure: 2 rows (true vs predicted) × (n_samples * T) columns
    fig, axes = plt.subplots(2, n_samples * T, figsize=(3*n_samples*T, 6))
    
    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)
    
    # Plot each sample
    for sample_idx, batch_idx in enumerate(selected_indices):
        for t in range(T):
            col_idx = sample_idx * T + t
            
            # Plot true frame (top row)
            axes[0, col_idx].imshow(true_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes[0, col_idx].set_title(f'Sample {sample_idx+1} - True Frame {t+1}')
            axes[0, col_idx].axis('off')
            
            # Plot predicted frame (bottom row)
            axes[1, col_idx].imshow(pred_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes[1, col_idx].set_title(f'Sample {sample_idx+1} - Predicted Frame {t+1}')
            axes[1, col_idx].axis('off')
    
    # Add overall title
    fig.suptitle(f'{model_name.replace("_", " ").title()} - Epoch {epoch} - Validation Samples', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name}_epoch_{epoch:03d}_combined.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved combined validation plot for epoch {epoch} to {filepath}")


def plot_mae_validation_samples(
    masked_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    true_frames: torch.Tensor,
    epoch: int,
    sample_indices: List[int],
    output_dir: str = "evaluation_plots/decoder_plots/mae_pretrain",
    model_name: str = "mae_pretrain"
) -> None:
    """
    Plot MAE pretraining validation samples with masked input, reconstruction, and ground truth.
    
    Args:
        masked_frames: Masked input frames [B, T, H, W]
        reconstructed_frames: Reconstructed frames [B, T, H, W] 
        true_frames: Ground truth frames [B, T, H, W]
        epoch: Current epoch number
        sample_indices: List of sample indices to plot (should be 5 samples)
        output_dir: Directory to save plots
        model_name: Name of the model for plot title
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up any images from future epochs
    cleanup_future_epoch_images(output_path, epoch, model_name)
    
    # Convert tensors to numpy and move to CPU
    masked_states = masked_frames.detach().cpu().numpy()
    reconstructed_states = reconstructed_frames.detach().cpu().numpy()
    true_states = true_frames.detach().cpu().numpy()
    
    # Get dimensions
    batch_size, T, H, W = true_states.shape
    
    # Select the 5 samples using provided indices
    n_samples = min(5, len(sample_indices), batch_size)
    selected_indices = sample_indices[:n_samples]
    
    # Plot each sample
    for sample_idx, batch_idx in enumerate(selected_indices):
        if sample_idx >= n_samples:
            break
            
        # Create a figure with 3 rows (masked, reconstructed, true) and T columns for time frames
        fig_sample, axes_sample = plt.subplots(3, T, figsize=(3*T, 9))
        
        # Handle case where T=1 to ensure axes is 2D
        if T == 1:
            axes_sample = axes_sample.reshape(3, 1)
        
        # Plot masked frames (top row)
        for t in range(T):
            axes_sample[0, t].imshow(masked_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes_sample[0, t].set_title(f'Masked Frame {t+1}')
            axes_sample[0, t].axis('off')
        
        # Plot reconstructed frames (middle row)
        for t in range(T):
            axes_sample[1, t].imshow(reconstructed_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes_sample[1, t].set_title(f'Reconstructed Frame {t+1}')
            axes_sample[1, t].axis('off')
            
        # Plot true frames (bottom row)
        for t in range(T):
            axes_sample[2, t].imshow(true_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes_sample[2, t].set_title(f'True Frame {t+1}')
            axes_sample[2, t].axis('off')
        
        # Add overall title
        fig_sample.suptitle(f'{model_name.replace("_", " ").title()} - Epoch {epoch} - Sample {sample_idx+1}', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        filename = f"{model_name}_epoch_{epoch:03d}_sample_{sample_idx+1}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_sample)
    
    print(f"Saved MAE validation plots for epoch {epoch} to {output_path}")


def plot_encoder_decoder_combined_validation_samples(
    current_states: torch.Tensor,
    predicted_next_states: torch.Tensor,
    true_next_states: torch.Tensor,
    epoch: int,
    sample_indices: List[int],
    output_dir: str = "evaluation_plots/decoder_plots/encoder_decoder",
    model_name: str = "encoder_decoder"
) -> None:
    """
    Plot encoder-decoder validation samples with current state, predicted next state, and true next state.
    
    Args:
        current_states: Current state frames [B, T, H, W]
        predicted_next_states: Predicted next state frames [B, T, H, W] 
        true_next_states: Ground truth next state frames [B, T, H, W]
        epoch: Current epoch number
        sample_indices: List of sample indices to plot (should be 3 samples)
        output_dir: Directory to save plots
        model_name: Name of the model for plot title
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up any images from future epochs
    cleanup_future_epoch_images(output_path, epoch, model_name)
    
    # Convert tensors to numpy and move to CPU
    print(f"   🔄 Processing data: Converting tensors to numpy...")
    current_states_np = current_states.detach().cpu().numpy()
    pred_states_np = predicted_next_states.detach().cpu().numpy()
    true_states_np = true_next_states.detach().cpu().numpy()
    
    # Convert from [0, 1] range back to [0, 255] for plotting
    current_states_np = (current_states_np * 255).astype('uint8')
    pred_states_np = (pred_states_np * 255).astype('uint8')
    true_states_np = (true_states_np * 255).astype('uint8')
    

    
    # Handle different tensor shapes
    if len(true_states_np.shape) == 4:
        # [B, T, H, W] format
        batch_size, T, H, W = true_states_np.shape
        use_rgb = False
    elif len(true_states_np.shape) == 5:
        # [B, T, C, H, W] format
        batch_size, T, C, H, W = true_states_np.shape
        if C == 3:
            # Keep RGB format for better visualization
            use_rgb = True
            # Don't convert to grayscale - keep RGB for better visibility
        else:
            # Take first channel for non-RGB data
            true_states_np = true_states_np[:, :, 0, :, :]
            current_states_np = current_states_np[:, :, 0, :, :]
            pred_states_np = pred_states_np[:, :, 0, :, :]
            use_rgb = False
    else:
        raise ValueError(f"Unexpected tensor shape: {true_states_np.shape}")
    

    

    

    

    
    # Select the 3 samples using provided indices, ensuring they're within bounds
    n_samples = min(3, len(sample_indices), batch_size)
    selected_indices = sample_indices[:n_samples]
    
    # Validate that all indices are within bounds
    valid_indices = []
    for idx in selected_indices:
        if idx < batch_size:
            valid_indices.append(idx)
        else:
            print(f"   ⚠️  Skipping invalid index {idx} (batch_size={batch_size})")
    
    if not valid_indices:
        print(f"   ❌ No valid indices found, skipping plot")
        return
    
    selected_indices = valid_indices
    n_samples = len(selected_indices)

    
    # Create figure: 3 rows (current, predicted, true) × (n_samples * T) columns
    fig, axes = plt.subplots(3, n_samples * T, figsize=(3*n_samples*T, 9))
    
    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)
    
    # Plot each sample

    for sample_idx, batch_idx in enumerate(selected_indices):
        for t in range(T):
            col_idx = sample_idx * T + t
            
            # Plot current state frame (top row)
            if use_rgb:
                # For RGB data, transpose from (C, H, W) to (H, W, C) for matplotlib
                current_frame = current_states_np[batch_idx, t].transpose(1, 2, 0)
                axes[0, col_idx].imshow(current_frame, vmin=0, vmax=VMAX)
            else:
                # For sparse data, use a different approach to make small values visible
                current_frame = current_states_np[batch_idx, t]
                if current_frame.max() < 10:  # Very sparse data
                    # Use log scale or different colormap to make small values visible
                    axes[0, col_idx].imshow(current_frame, cmap='hot', vmin=0, vmax=current_frame.max() if current_frame.max() > 0 else 1)
                else:
                    axes[0, col_idx].imshow(current_frame, cmap='gray', vmin=0, vmax=VMAX)
            axes[0, col_idx].set_title(f'Sample {sample_idx+1} - Current Frame {t+1}')
            axes[0, col_idx].axis('off')
            
            # Plot predicted next state frame (middle row)
            if use_rgb:
                pred_frame = pred_states_np[batch_idx, t].transpose(1, 2, 0)
                # For predicted frames, use adaptive normalization to make small values visible
                if pred_frame.max() < 50:  # Very small values
                    # Use multiple approaches to make small values visible
                    if pred_frame.max() < 10:  # Extremely small values
                        # Use log scaling for very dark images
                        pred_frame_log = np.log1p(pred_frame * 25)  # log(1 + x*25) to make small values visible
                        pred_frame_log_normalized = pred_frame_log / pred_frame_log.max() * 255
                        axes[1, col_idx].imshow(pred_frame_log_normalized.astype('uint8'), cmap='hot', vmin=0, vmax=255)
    
                    else:
                        # Normalize to make small values more visible
                        pred_frame_normalized = pred_frame.astype(float) / max(pred_frame.max(), 1e-6) * 255
                        axes[1, col_idx].imshow(pred_frame_normalized.astype('uint8'), vmin=0, vmax=255)

                else:
                    axes[1, col_idx].imshow(pred_frame, vmin=0, vmax=VMAX)
            else:
                # For sparse data, use a different approach to make small values visible
                pred_frame = pred_states_np[batch_idx, t]
                if pred_frame.max() < 10:  # Very sparse data
                    # Use log scale or different colormap to make small values visible
                    axes[1, col_idx].imshow(pred_frame, cmap='hot', vmin=0, vmax=pred_frame.max() if pred_frame.max() > 0 else 1)
                else:
                    axes[1, col_idx].imshow(pred_frame, cmap='gray', vmin=0, vmax=VMAX)
            axes[1, col_idx].set_title(f'Sample {sample_idx+1} - Predicted Next Frame {t+1}')
            axes[1, col_idx].axis('off')
            
            # Plot true next state frame (bottom row)
            if use_rgb:
                true_frame = true_states_np[batch_idx, t].transpose(1, 2, 0)
                axes[2, col_idx].imshow(true_frame, vmin=0, vmax=VMAX)
            else:
                # For sparse data, use a different approach to make small values visible
                true_frame = true_states_np[batch_idx, t]
                if true_frame.max() < 10:  # Very sparse data
                    # Use log scale or different colormap to make small values visible
                    axes[2, col_idx].imshow(true_frame, cmap='hot', vmin=0, vmax=true_frame.max() if true_frame.max() > 0 else 1)
                else:
                    axes[2, col_idx].imshow(true_frame, cmap='gray', vmin=0, vmax=VMAX)
            axes[2, col_idx].set_title(f'Sample {sample_idx+1} - True Next Frame {t+1}')
            axes[2, col_idx].axis('off')
    
    # Add overall title
    fig.suptitle(f'{model_name.replace("_", " ").title()} - Epoch {epoch} - Current → Predicted → True', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name}_epoch_{epoch:03d}_current_predicted_true.png"
    filepath = output_path / filename
    print(f"   💾 Saving plot to: {filepath}")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✅ Plot saved successfully!")


def plot_mae_combined_validation_samples(
    masked_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    true_frames: torch.Tensor,
    epoch: int,
    sample_indices: List[int],
    output_dir: str = "evaluation_plots/decoder_plots/mae_pretrain",
    model_name: str = "mae_pretrain"
) -> None:
    """
    Plot all 5 MAE samples in a single combined figure.
    
    Args:
        masked_frames: Masked input frames [B, T, H, W]
        reconstructed_frames: Reconstructed frames [B, T, H, W] 
        true_frames: Ground truth frames [B, T, H, W]
        epoch: Current epoch number
        sample_indices: List of sample indices to plot (should be 5 samples)
        output_dir: Directory to save plots
        model_name: Name of the model for plot title
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up any images from future epochs
    cleanup_future_epoch_images(output_path, epoch, model_name)
    
    # Convert tensors to numpy and move to CPU
    masked_states = masked_frames.detach().cpu().numpy()
    reconstructed_states = reconstructed_frames.detach().cpu().numpy()
    true_states = true_frames.detach().cpu().numpy()
    
    # Convert from [0, 1] range back to [0, 255] for plotting
    masked_states = (masked_states * 255).astype('uint8')
    reconstructed_states = (reconstructed_states * 255).astype('uint8')
    true_states = (true_states * 255).astype('uint8')
    
    # Get dimensions
    batch_size, T, H, W = true_states.shape
    
    # Select the 5 samples using provided indices
    n_samples = min(5, len(sample_indices), batch_size)
    selected_indices = sample_indices[:n_samples]
    
    # Create figure: 3 rows (masked, reconstructed, true) × (n_samples * T) columns
    fig, axes = plt.subplots(3, n_samples * T, figsize=(3*n_samples*T, 9))
    
    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)
    
    # Plot each sample
    for sample_idx, batch_idx in enumerate(selected_indices):
        for t in range(T):
            col_idx = sample_idx * T + t
            
            # Plot masked frame (top row)
            axes[0, col_idx].imshow(masked_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes[0, col_idx].set_title(f'Sample {sample_idx+1} - Masked Frame {t+1}')
            axes[0, col_idx].axis('off')
            
            # Plot reconstructed frame (middle row)
            axes[1, col_idx].imshow(reconstructed_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes[1, col_idx].set_title(f'Sample {sample_idx+1} - Reconstructed Frame {t+1}')
            axes[1, col_idx].axis('off')
            
            # Plot true frame (bottom row)
            axes[2, col_idx].imshow(true_states[batch_idx, t], cmap='gray', vmin=0, vmax=VMAX)
            axes[2, col_idx].set_title(f'Sample {sample_idx+1} - True Frame {t+1}')
            axes[2, col_idx].axis('off')
    
    # Add overall title
    fig.suptitle(f'{model_name.replace("_", " ").title()} - Epoch {epoch} - Validation Samples', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name}_epoch_{epoch:03d}_combined.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved MAE combined validation plot for epoch {epoch} to {filepath}")


def get_random_validation_samples(val_dataloader, n_samples: int = 5, seed: Optional[int] = None) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get random samples from validation dataloader for consistent plotting across epochs.
    
    Args:
        val_dataloader: Validation data loader
        n_samples: Number of samples to select (default: 5)
        seed: Random seed for reproducible sample selection
        
    Returns:
        Tuple of (sample_indices, states, next_states, actions) for the selected samples
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Get a batch from validation data
    val_iter = iter(val_dataloader)
    batch = next(val_iter)
    state, next_state, action, _ = batch
    
    batch_size = state.shape[0]
    
    # Select random indices, but ensure we don't exceed batch size
    available_indices = list(range(batch_size))
    n_samples = min(n_samples, batch_size)  # Ensure we don't try to select more than available
    sample_indices = random.sample(available_indices, n_samples)
    

    
    return sample_indices, state[sample_indices], next_state[sample_indices], action[sample_indices]


def should_plot_validation(epoch: int, plot_frequency: int = 5) -> bool:
    """
    Determine if validation plots should be generated for this epoch.
    
    Args:
        epoch: Current epoch number (0-indexed)
        plot_frequency: How often to generate plots (every N epochs)
        
    Returns:
        True if plots should be generated for this epoch
    """
    return (epoch + 1) % plot_frequency == 0