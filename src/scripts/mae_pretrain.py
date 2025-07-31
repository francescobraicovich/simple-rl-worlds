#!/usr/bin/env python3
"""
MAE Pretraining Script

This script implements self-supervised pretraining of Vision Transformer encoder and 
decoder models using Masked Autoencoder (MAE) approach. The main objective is to train:
1. An encoder to produce powerful visual representations from unmasked patches
2. A decoder to reconstruct masked patches from encoded representations

The training uses masked reconstruction loss where a large portion of image patches
are masked and the model learns to reconstruct them from the visible patches.
"""

import os
# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.optim as optim
import wandb

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_decoder, load_config
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device
from src.utils.scheduler_utils import create_lr_scheduler, step_scheduler
from src.utils.masking import random_masking, patchify, compute_reconstruction_loss, unpatchify, get_masked_image
from src.utils.plot import plot_mae_validation_samples


class MAETrainer:
    """
    Trainer class for Masked Autoencoder (MAE) pretraining.
    
    This class handles the complete pretraining pipeline including:
    - Model initialization (encoder and decoder)
    - Data loading and preprocessing
    - Masked reconstruction training and validation loops
    - Random patch masking and reconstruction loss computation
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the MAE trainer.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Training parameters from pretraining section
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
        self.encoder = None
        self.decoder = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/mae_pretraining")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Step counter for batch-level logging
        self.global_step = 0
        
        # Plotting configuration
        self.plot_frequency = 1  # Plot every epoch for MAE
        self.plot_dir = "evaluation_plots/decoder_plots/mae_pretrain"
        self.val_sample_indices = None  # Will be set once during first validation
        
    def compute_gradient_vanishing_ratio(self) -> float:
        """
        Compute gradient vanishing ratio as the ratio of small gradients to total gradients.
        
        Returns:
            Ratio of gradients with magnitude < 1e-7 to total gradients
        """
        total_params = 0
        small_grad_params = 0
        
        for model in [self.encoder, self.decoder]:
            for param in model.parameters():
                if param.grad is not None:
                    grad_magnitude = param.grad.abs()
                    total_params += grad_magnitude.numel()
                    small_grad_params += (grad_magnitude < 1e-7).sum().item()
        
        if total_params == 0:
            return 0.0
        return small_grad_params / total_params
    
    def plot_validation_samples(self, epoch: int):
        """
        Generate and save validation plots showing masked input, reconstruction, and ground truth.
        
        Args:
            epoch: Current epoch number
        """
        if self.val_dataloader is None:
            print(f"Skipping validation plots for epoch {epoch}: No validation data available")
            return
            
        try:
            self.encoder.eval()
            self.decoder.eval()
            
            with torch.no_grad():
                # Get a batch from validation data
                val_iter = iter(self.val_dataloader)
                batch = next(val_iter)
                state, _, _, _ = batch
                
                # Move to device
                state = state.to(self.device)
                batch_size = state.shape[0]
                
                # Select sample indices for consistent plotting (first time only)
                if self.val_sample_indices is None:
                    n_samples = min(5, batch_size)
                    self.val_sample_indices = list(range(n_samples))
                
                # Select only the samples we want to plot
                plot_states = state[self.val_sample_indices]
                
                # 1. Patch embedding
                x = self.encoder.patch_embed(plot_states)  # [B, N, D]
                
                # 2. Random masking (use same seed for consistent masking)
                torch.manual_seed(1)  # For consistent masking across epochs
                x_vis, mask, ids_keep, ids_restore = random_masking(x, self.mask_ratio)
                
                # 3. Encode visible patches only
                x_encoded = self.encoder(plot_states, ids_keep)  # [B, N_vis+1, D]
                
                # Remove CLS token for decoder
                x_encoded = x_encoded[:, 1:, :]  # [B, N_vis, D]
                
                # 4. Decode to reconstruct all patches
                x_reconstructed = self.decoder(x_encoded, ids_restore)  # [B, N, P*P*C]
                
                # 5. Convert patches back to images for visualization
                # Get the ground truth images
                true_frames = plot_states  # [B, C, H, W]
                
                # Convert reconstructed patches to images
                reconstructed_frames = unpatchify(
                    x_reconstructed, 
                    self.encoder.config.patch_size, 
                    self.encoder.config.image_size
                )  # [B, C, H, W]
                
                # Create masked images for visualization
                masked_frames = get_masked_image(
                    plot_states, 
                    mask, 
                    self.encoder.config.patch_size
                )  # [B, C, H, W]
                
                # Convert to proper format for plotting (take first channel if stacked)
                #if true_frames.shape[1] > 1:  # If we have stacked frames
                #    true_frames = true_frames[:, 0:1, :, :]  # Take first channel
                #    reconstructed_frames = reconstructed_frames[:, 0:1, :, :]
                #    masked_frames = masked_frames[:, 0:1, :, :]
                
                # Plot the samples
                plot_mae_validation_samples(
                    masked_frames=masked_frames,
                    reconstructed_frames=reconstructed_frames,
                    true_frames=true_frames,
                    epoch=epoch,
                    sample_indices=list(range(len(self.val_sample_indices))),
                    output_dir=self.plot_dir,
                    model_name="mae_pretrain"
                )
                
                print(f"Generated validation plots for epoch {epoch}")
                
        except Exception as e:
            print(f"Warning: Failed to generate validation plots for epoch {epoch}: {e}")
        
    def initialize_models(self):
        """Initialize encoder and decoder models."""
        # Initialize models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.decoder = init_decoder(self.config_path).to(self.device)
        
        print(f"Initialized encoder with {sum(p.numel() for p in self.encoder.parameters()):,} parameters")
        print(f"Initialized decoder with {sum(p.numel() for p in self.decoder.parameters()):,} parameters")
        
    def initialize_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        # Combine all trainable parameters
        all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        print(f"Initialized AdamW optimizer with lr={self.learning_rate}, weight_decay={self.weight_decay}")
        
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
        # Initialize data loading pipeline (loads existing data only)
        pipeline = DataLoadingPipeline(
            batch_size=self.batch_size,
            config_path=self.config_path
        )

        # Run the pipeline to get dataloaders from existing data
        self.train_dataloader, self.val_dataloader = pipeline.run_pipeline()
        
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
        """
        Perform a single MAE training step.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Tuple of (loss value, gradient vanishing ratio) for this batch
        """
        state, _, _, _ = batch  # Use current state for reconstruction
        
        # Move to device
        state = state.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # 1. Patch embedding (encoder's first step)
        x = self.encoder.patch_embed(state)  # [B, N, D]
        
        # 2. Random masking
        x_vis, mask, ids_keep, ids_restore = random_masking(x, self.mask_ratio)
        
        # 3. Encode visible patches only
        x_encoded = self.encoder(state, ids_keep)  # [B, N_vis+1, D] (includes CLS token)
        
        # Remove CLS token for decoder
        x_encoded = x_encoded[:, 1:, :]  # [B, N_vis, D]
        
        # 4. Decode to reconstruct all patches
        x_reconstructed = self.decoder(x_encoded, ids_restore)  # [B, N, P*P*C]
        
        # 5. Compute reconstruction loss on masked patches only
        # Create target patches from original images
        target_patches = patchify(state, self.encoder.config.patch_size)  # [B, N, P*P*C]
        
        # Compute loss only on masked patches
        loss = compute_reconstruction_loss(x_reconstructed, target_patches, mask)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient vanishing ratio before clipping
        grad_vanishing_ratio = self.compute_gradient_vanishing_ratio()
        
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            trainable_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clipping)
        
        self.optimizer.step()
        
        # Log batch metrics to wandb
        self.global_step += 1
        if wandb.run is not None:
            wandb.log({
                "batch_loss": loss.item(),
                "gradient_vanishing_ratio": grad_vanishing_ratio,
                "global_step": self.global_step
            })
        
        return loss.item(), grad_vanishing_ratio
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Loss value for this batch
        """
        state, _, _, _ = batch  # Use current state for reconstruction
        
        # Move to device
        state = state.to(self.device)
        
        with torch.no_grad():
            # 1. Patch embedding
            x = self.encoder.patch_embed(state)  # [B, N, D]
            
            # 2. Random masking
            x_vis, mask, ids_keep, ids_restore = random_masking(x, self.mask_ratio)
            
            # 3. Encode visible patches only
            x_encoded = self.encoder(state, ids_keep)  # [B, N_vis+1, D]
            
            # Remove CLS token for decoder
            x_encoded = x_encoded[:, 1:, :]  # [B, N_vis, D]
            
            # 4. Decode to reconstruct all patches
            x_reconstructed = self.decoder(x_encoded, ids_restore)  # [B, N, P*P*C]
            
            # 5. Compute reconstruction loss on masked patches only
            target_patches = patchify(state, self.encoder.config.patch_size)  # [B, N, P*P*C]
            loss = compute_reconstruction_loss(x_reconstructed, target_patches, mask)
        
        return loss.item()
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average training loss, average gradient vanishing ratio) for the epoch
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        total_grad_vanishing = 0.0
        num_batches = 0
        
        for batch in self.train_dataloader:
            loss, grad_vanishing_ratio = self.train_step(batch)
            total_loss += loss
            total_grad_vanishing += grad_vanishing_ratio
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_vanishing = total_grad_vanishing / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_grad_vanishing
        
    def validate_epoch(self) -> Optional[float]:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss for the epoch, or None if no validation data
        """
        if self.val_dataloader is None:
            return None
            
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch (can be None)
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
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
        
        # Save best checkpoint based on validation loss (or training loss if no validation)
        current_loss = val_loss if val_loss is not None else train_loss
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"New best checkpoint saved with loss: {current_loss:.6f}")
            
    def train(self):
        """
        Run the complete MAE pretraining process.
        """
        print("="*60)
        print("Starting MAE Pretraining")
        print("="*60)
        
        # Initialize everything
        self.initialize_models()
        self.initialize_optimizer()
        self.load_data()
        
        # Initialize wandb if configured
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'mae-pretraining'),
                entity=wandb_config.get('entity'),
                name=f"mae-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, avg_grad_vanishing = self.train_epoch()

            # Validate
            val_loss = self.validate_epoch()

            epoch_time = time.time() - epoch_start_time
            
            # Log to wandb
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "avg_gradient_vanishing_ratio": avg_grad_vanishing,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
                
            if wandb.run is not None:
                wandb.log(log_dict)
                
            # Terminal output (minimal)
            if val_loss is not None:
                # Base message
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Grad Vanishing: {avg_grad_vanishing:.4f}"
                print(message)
            else:
                # Base message for no validation
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: N/A, Grad Vanishing: {avg_grad_vanishing:.4f}"
                print(message)
                
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss)
            
            # Generate validation plots
            if (epoch + 1) % self.plot_frequency == 0:
                self.plot_validation_samples(epoch)
            
            # Step learning rate scheduler
            if self.lr_scheduler is not None:
                # Use validation loss for plateau scheduler, otherwise step normally
                step_scheduler(self.lr_scheduler, val_loss)
            
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MAE pretraining models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = MAETrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
