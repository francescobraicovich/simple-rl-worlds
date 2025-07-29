#!/usr/bin/env python3
"""
Vision Transformer Masked Autoencoder (ViTMAE) Pretraining Script

This script implements self-supervised pretraining using Hugging Face's Vision Transformer
Masked Autoencoder approach. The main objective is to train the ViTMAE model to reconstruct
masked patches of input images, learning powerful visual representations.

The training process:
1. Take input video frames and convert them to patches
2. Randomly mask a subset of patches
3. Encode visible patches through the ViT encoder
4. Decode latent representations to reconstruct masked patches
5. Compute reconstruction loss only on masked patches
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.optim as optim
import wandb

# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_vit_mae, load_config
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device
from src.utils.scheduler_utils import create_lr_scheduler, step_scheduler, get_current_lr


class ViTMAETrainer:
    """
    Trainer class for Vision Transformer Masked Autoencoder (ViTMAE) pretraining.
    
    This class handles the complete pretraining pipeline including:
    - ViTMAE model initialization
    - Data loading and preprocessing
    - Patch masking and reconstruction
    - Training and validation loops with reconstruction loss
    - Optimization of model parameters
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the ViTMAE trainer.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Training parameters
        self.training_config = self.config['training']['pretraining']
        self.num_epochs = self.training_config['num_epochs']
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = self.training_config['learning_rate']
        self.weight_decay = self.training_config['weight_decay']
        self.gradient_clipping = self.training_config.get('gradient_clipping', None)
        self.mask_ratio = self.training_config['mask_ratio']
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models
        self.vit_mae = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None

        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/vitmae_pretraining")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Plotting configuration
        self.plot_frequency = 1  # Plot every epoch
        self.plot_dir = "evaluation_plots/decoder_plots/vitmae_pretrain"
        self.validation_sample_indices = None  # Will be set once data is loaded
        
        # Gradient statistics tracking
        self.last_epoch_grad_stats = {}
        
    def initialize_models(self):
        """Initialize ViTMAE model."""
        # Initialize ViTMAE model
        self.vit_mae = init_vit_mae(self.config_path).to(self.device)
        
        print(f"Initialized ViTMAE with {sum(p.numel() for p in self.vit_mae.parameters())} parameters")
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer and learning rate scheduler for all trainable parameters."""
        # Get all trainable parameters from ViTMAE model
        trainable_params = list(self.vit_mae.parameters())
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize learning rate scheduler if configured
        scheduler_config = self.training_config.get('lr_scheduler', {})
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer, 
            scheduler_config, 
            self.num_epochs
        )
        
        if self.lr_scheduler is not None:
            print(f"Initialized {scheduler_config.get('type', 'cosine')} learning rate scheduler")
        else:
            print("No learning rate scheduler configured")
            
    def load_data(self):
        """Load training and validation data using DataLoadingPipeline."""
        data_pipeline = DataLoadingPipeline(
            self.batch_size,
            self.config_path
        )

        # Get training and validation dataloaders
        self.train_dataloader, self.val_dataloader = data_pipeline.run_pipeline()
        
        print(f"Loaded {len(self.train_dataloader)} training batches")
        if self.val_dataloader is not None:
            print(f"Loaded {len(self.val_dataloader)} validation batches")
        else:
            print("No validation data available")
    
    def preprocess_frames_for_vitmae(self, state: torch.Tensor) -> torch.Tensor:
        """
        Preprocess video frames for ViTMAE input.
        
        Args:
            state: Input tensor of shape [B, T, H, W] (batch, time, height, width)
            
        Returns:
            Preprocessed tensor of shape [B*T, C, H, W] where C=4 (stacked frames as channels)
        """
        B, T, H, W = state.shape
        
        # Reshape to treat each sequence as a multi-channel image: [B, T, H, W] -> [B, T, H, W]
        # We'll treat the temporal dimension as channels
        frames = state  # Keep as [B, T, H, W]
        
        # Normalize pixel values to [0, 1] if needed
        if frames.max() > 1.0:
            frames = frames / 255.0
            
        return frames
    
    def create_masked_visualization(self, pixel_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Create a visualization of the masked input by setting masked patches to zero.
        
        Args:
            pixel_values: Input frames [B, T, H, W]
            mask: ViTMAE mask [B, sequence_length] where 1=masked, 0=visible
            
        Returns:
            Masked frames with masked patches set to zero [B, T, H, W]
        """
        B, T, H, W = pixel_values.shape
        
        # Get patch size from ViTMAE config (7x7 patches for 84x84 images)
        patch_size = 7  # From the ViTMAE config
        
        # Calculate number of patches (84/7 = 12 patches per side)
        num_patches_h = H // patch_size  # 12
        num_patches_w = W // patch_size  # 12
        
        # Create a copy of the input frames
        masked_frames = pixel_values.clone()
        
        # Reshape mask to match patch layout: [B, 144] -> [B, 12, 12]
        mask_2d = mask.view(B, num_patches_h, num_patches_w)
        
        # Apply mask by setting masked patches to zero
        for b in range(B):
            for h_patch in range(num_patches_h):
                for w_patch in range(num_patches_w):
                    if mask_2d[b, h_patch, w_patch] == 1:  # If this patch is masked
                        # Set all pixels in this patch to zero for all time frames
                        h_start = h_patch * patch_size
                        h_end = h_start + patch_size
                        w_start = w_patch * patch_size
                        w_end = w_start + patch_size
                        masked_frames[b, :, h_start:h_end, w_start:w_end] = 0.0
        
        return masked_frames
    
    def unpatchify_vitmae_output(self, logits: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """
        Convert ViTMAE logits back to image format.
        
        Args:
            logits: ViTMAE output logits [B, sequence_length, patch_size^2 * num_channels]
            original_shape: Original input shape [B, T, H, W]
            
        Returns:
            Reconstructed images [B, T, H, W]
        """
        B, T, H, W = original_shape
        
        # Get patch size from ViTMAE config (7x7 patches for 84x84 images)
        patch_size = 7  # From the ViTMAE config
        
        # Calculate number of patches (84/7 = 12 patches per side)
        num_patches_h = H // patch_size  # 12
        num_patches_w = W // patch_size  # 12
        
        # logits shape: [B, 144, 49*4] = [B, 144, 196]
        # where 144 = 12*12 patches, 196 = 7*7*4 (patch_size^2 * num_channels)
        
        # Reshape logits: [B, 144, 196] -> [B, 12, 12, 7, 7, 4]
        logits = logits.view(B, num_patches_h, num_patches_w, patch_size, patch_size, T)
        
        # Rearrange to get final image: [B, 4, 84, 84] -> [B, T, H, W]
        reconstructed = logits.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(B, T, H, W)
        
        return reconstructed
    
    def compute_gradient_vanishing_ratio(self) -> dict:
        """
        Compute gradient vanishing ratio for monitoring gradient flow.
        
        Returns:
            Dictionary containing gradient statistics
        """
        total_norm = 0.0
        param_count = 0
        layer_norms = {}
        
        for name, param in self.vit_mae.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Track layer-specific norms
                layer_name = name.split('.')[0]  # Get the first part of the parameter name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm.item())
        
        total_norm = total_norm ** (1. / 2)
        
        # Compute layer-wise gradient norms
        layer_stats = {}
        for layer_name, norms in layer_norms.items():
            layer_stats[f"grad_norm_{layer_name}"] = sum(norms) / len(norms) if norms else 0.0
        
        # Compute gradient vanishing ratio (ratio of smallest to largest layer norm)
        layer_means = [stats for stats in layer_stats.values() if stats > 0]
        vanishing_ratio = min(layer_means) / max(layer_means) if len(layer_means) > 1 else 1.0
        
        return {
            "gradient_norm_total": total_norm,
            "gradient_vanishing_ratio": vanishing_ratio,
            **layer_stats
        }

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single training step with ViTMAE reconstruction loss.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Reconstruction loss for this batch
        """
        state, next_state, action, reward = batch

        # Move to device - we use current state for ViTMAE pretraining
        state = state.to(self.device)  # [B, T, H, W]
        
        # Preprocess frames for ViTMAE
        pixel_values = self.preprocess_frames_for_vitmae(state)  # [B, T, H, W]
        
        # Forward pass through ViTMAE
        outputs = self.vit_mae(pixel_values=pixel_values)
        
        # Get reconstruction loss from ViTMAE
        loss = outputs.loss
        
        # Backward pass and optimizer step
        loss.backward()
        
        # Compute gradient statistics before clipping
        grad_stats = self.compute_gradient_vanishing_ratio()
        
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.vit_mae.parameters(), self.gradient_clipping)
            
        self.optimizer.step()
        self.optimizer.zero_grad()  # Reset gradients for next step
        
        return loss.item(), grad_stats
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step with ViTMAE reconstruction loss.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Reconstruction loss for this batch
        """
        state, next_state, action, reward = batch

        # Move to device - we use current state for ViTMAE pretraining
        state = state.to(self.device)  # [B, T, H, W]
        
        with torch.no_grad():
            # Preprocess frames for ViTMAE
            pixel_values = self.preprocess_frames_for_vitmae(state)  # [B, T, H, W]
            
            # Forward pass through ViTMAE
            outputs = self.vit_mae(pixel_values=pixel_values)
            
            # Get reconstruction loss from ViTMAE
            loss = outputs.loss
            
        return loss.item()
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.vit_mae.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Accumulate gradient statistics across batches
        epoch_grad_stats = {}
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, grad_stats = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Accumulate gradient statistics
            for key, value in grad_stats.items():
                if key not in epoch_grad_stats:
                    epoch_grad_stats[key] = []
                epoch_grad_stats[key].append(value)
            
            # Log batch-level metrics to wandb if enabled
            if self.config['wandb']['enabled']:
                log_dict = {
                    "batch_loss": loss,
                    "batch": batch_idx,
                    **grad_stats  # Include gradient statistics
                }
                wandb.log(log_dict)
                
        # Compute average gradient statistics for the epoch
        avg_grad_stats = {}
        for key, values in epoch_grad_stats.items():
            avg_grad_stats[f"epoch_{key}"] = sum(values) / len(values) if values else 0.0
        
        # Store epoch gradient stats for logging in main training loop
        self.last_epoch_grad_stats = avg_grad_stats
        
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def validate_epoch(self) -> Optional[float]:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss for the epoch, or None if no validation data
        """
        if self.val_dataloader is None:
            return None
            
        self.vit_mae.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def plot_vitmae_validation_predictions(self, epoch: int):
        """Generate ViTMAE validation plots for the current epoch if needed."""
        # Import here to avoid issues with module-level imports
        try:
            from src.utils.plot import plot_mae_validation_samples, get_random_validation_samples, should_plot_validation
        except ImportError:
            # If plotting utilities are not available, skip plotting
            return
        
        # Check if we should plot for this epoch
        if not should_plot_validation(epoch, self.plot_frequency):
            return
            
        if self.val_dataloader is None:
            return
            
        # Get validation sample indices (consistent across epochs)
        if self.validation_sample_indices is None:
            self.validation_sample_indices, _, _, _ = get_random_validation_samples(
                self.val_dataloader, n_samples=5, seed=42
            )
        
        # Set model to eval mode
        self.vit_mae.eval()
        
        # Get a validation batch
        val_iter = iter(self.val_dataloader)
        batch = next(val_iter)
        state, next_state, action, _ = batch
        
        # Move to device - we use current state for ViTMAE pretraining
        state = state.to(self.device)  # [B, T, H, W]
        
        with torch.no_grad():
            # Preprocess frames for ViTMAE
            pixel_values = self.preprocess_frames_for_vitmae(state)  # [B, T, H, W]
            
            # Forward pass through ViTMAE to get reconstructions
            outputs = self.vit_mae(pixel_values=pixel_values)
            
            # Get reconstructed images from ViTMAE output
            # ViTMAE returns logits that need to be converted back to image format
            logits = outputs.logits  # [B, sequence_length, patch_size^2 * num_channels]
            mask = outputs.mask      # [B, sequence_length] - 1 for masked, 0 for visible
            
            # Convert logits back to image format using the model's unpatchify method
            try:
                # Try to use the model's built-in unpatchify method if available
                reconstructed = self.vit_mae.unpatchify(logits)  # Should return [B, C, H, W]
                # Reshape to match our expected format [B, T, H, W]
                if reconstructed.dim() == 4 and reconstructed.shape[1] == pixel_values.shape[1]:
                    reconstructed = reconstructed  # Already in [B, T, H, W] format
                else:
                    # If it's in a different format, reshape appropriately
                    reconstructed = reconstructed.view(state.shape)
            except (AttributeError, RuntimeError):
                # Fall back to our custom unpatchify method
                reconstructed = self.unpatchify_vitmae_output(logits, state.shape)  # [B, T, H, W]
            
            # Create masked input visualization by zeroing out masked patches
            masked_input = self.create_masked_visualization(pixel_values, mask)  # [B, T, H, W]
        
        # Generate plots using existing MAE plotting function
        # Show: masked input -> reconstructed -> original (true)
        try:
            plot_mae_validation_samples(
                masked_frames=masked_input,  # Show the actual masked input with zero patches
                reconstructed_frames=reconstructed,
                true_frames=state,    # Ground truth frames
                epoch=epoch,
                sample_indices=self.validation_sample_indices,
                output_dir=self.plot_dir,
                model_name="vitmae_pretrain"
            )
        except Exception as e:
            print(f"Plotting failed: {e}")
        
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch (None if no validation data)
        """
        checkpoint = {
            'epoch': epoch,
            'vitmae_state_dict': self.vit_mae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Use validation loss for best model selection, fallback to training loss
        current_val_loss = val_loss if val_loss is not None else train_loss
        
        # Save best checkpoint if validation loss improved
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {current_val_loss:.6f}")
                        
    def train(self):
        """
        Run the complete training loop.
        """
        print("🚀 Starting ViTMAE pretraining...")
        print(f"Configuration: {self.num_epochs} epochs, batch size {self.batch_size}, lr {self.learning_rate}")
        print(f"Mask ratio: {self.mask_ratio}, Device: {self.device}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.vit_mae.parameters())}")
        
        # Initialize wandb if enabled
        if self.config['wandb']['enabled']:
            wandb.init(
                project=self.config['wandb']['project'],
                name=f"vitmae_pretrain_{time.strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
            
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate_epoch()
            
            # Generate ViTMAE validation plots if needed
            self.plot_vitmae_validation_predictions(epoch)
            
            # Learning rate scheduling
            if self.lr_scheduler is not None:
                step_scheduler(self.lr_scheduler, val_loss)
                
            # Get current learning rate
            current_lr = get_current_lr(self.optimizer)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            val_loss_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.6f}{val_loss_str}, "
                  f"LR: {current_lr:.8f}, Time: {epoch_time:.2f}s")
            
            # Log epoch-level metrics to wandb if enabled
            if self.config['wandb']['enabled']:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                    **self.last_epoch_grad_stats  # Include epoch gradient statistics
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, train_loss, val_loss)
            
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        if self.config['wandb']['enabled']:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    
    parser = argparse.ArgumentParser(description='Train ViTMAE model for video pretraining')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: project_root/config.yaml)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ViTMAETrainer(config_path=args.config)
    
    # Initialize models and data
    trainer.initialize_models()
    trainer.initialize_optimizer()
    trainer.load_data()
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
