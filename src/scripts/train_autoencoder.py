#!/usr/bin/env python3
"""
Autoencoder Training Script

This script implements end-to-end training of an autoencoder:
encoder and decoder only. The main objective is to train both components
jointly to reconstruct the input state frame from its latent representation.

The training process:
1. Encode current state to latent representation
2. Decode latent representation to reconstruct the same frame
3. Optimize L1 reconstruction loss through the encoder-decoder graph
"""

# =============================================================================
# CONFIGURATION SECTION - Modify these settings as needed
# =============================================================================

AUTOENCODER_CONFIG = {
    # Training parameters
    'num_epochs': 1000,
    'batch_size': 32,
    'learning_rate': 0.002,
    'weight_decay': 0.0001,
    'gradient_clipping': 1.0,
    
    # Early stopping
    'early_stopping_enabled': False,
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.001,
    'restore_best_weights': True,
    
    # Learning rate scheduler
    'lr_scheduler_enabled': True,
    'lr_scheduler_type': 'cosine',
    'cosine_T_max': 50,
    'cosine_eta_min': 0.0001,
    
    # Wandb configuration
    'wandb_enabled': True,
    'wandb_project': 'autoencoder-training',
    'wandb_entity': None,
    
    # Output directories
    'checkpoint_dir': 'weights/autoencoder',
    'plot_output_dir': 'evaluation_plots/decoder_plots/autoencoder',
    
    # Validation
    'num_validation_samples': 3,
    'validation_seed': 42,
}

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

import os
# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add plotting imports
from src.utils.plot import plot_encoder_decoder_combined_validation_samples, get_random_validation_samples
from src.utils.init_models import init_encoder, init_decoder, load_config
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device
from src.utils.scheduler_utils import create_lr_scheduler, step_scheduler, get_current_lr


class AutoencoderTrainer:
    """
    Trainer class for end-to-end Encoder-Decoder autoencoder architecture.
    
    This class handles the complete training pipeline including:
    - Model initialization (encoder, decoder)
    - Data loading and preprocessing
    - Training and validation loops with reconstruction loss
    - Joint optimization of encoder and decoder components
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Autoencoder trainer.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Use local configuration
        self.autoencoder_config = AUTOENCODER_CONFIG
        
        # Training parameters
        self.num_epochs = self.autoencoder_config['num_epochs']
        self.batch_size = self.autoencoder_config['batch_size']
        self.learning_rate = self.autoencoder_config['learning_rate']
        self.weight_decay = self.autoencoder_config['weight_decay']
        self.gradient_clipping = self.autoencoder_config['gradient_clipping']
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models
        self.encoder = None
        self.decoder = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = nn.MSELoss()  # MSE loss for logits (no sigmoid)
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(self.autoencoder_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping configuration
        self.early_stopping_enabled = self.autoencoder_config['early_stopping_enabled']
        self.early_stopping_patience = self.autoencoder_config['early_stopping_patience']
        self.early_stopping_min_delta = self.autoencoder_config['early_stopping_min_delta']
        self.restore_best_weights = self.autoencoder_config['restore_best_weights']
        
        # Early stopping state
        self.epochs_without_improvement = 0
        self.best_weights = None
        
        # Plotting configuration
        self.plot_output_dir = self.autoencoder_config['plot_output_dir']
        self.validation_sample_indices = None
        self.validation_samples = None
        
        # Training progress tracking
        self.current_epoch = 0
        self.total_batches = 0
        self.batches_completed = 0
        
    def initialize_models(self):
        """Initialize encoder and decoder models."""
        # Initialize models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.decoder = init_decoder(self.config_path).to(self.device)
        
        # Print model parameter counts
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        # Model parameters logged silently
        pass
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer and learning rate scheduler for all trainable parameters."""
        # Combine parameters from encoder and decoder
        trainable_params = (list(self.encoder.parameters()) + 
                          list(self.decoder.parameters()))
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize learning rate scheduler if configured
        if self.autoencoder_config['lr_scheduler_enabled']:
            scheduler_config = {
                'enabled': True,
                'type': self.autoencoder_config['lr_scheduler_type'],
                'cosine_T_max': self.autoencoder_config['cosine_T_max'],
                'cosine_eta_min': self.autoencoder_config['cosine_eta_min']
            }
            self.lr_scheduler = create_lr_scheduler(
                self.optimizer, 
                scheduler_config, 
                self.num_epochs
            )
        else:
            self.lr_scheduler = None
        
        # Learning rate scheduler initialized silently
        pass
        
    def load_data(self):
        """Load training and validation data using DataLoadingPipeline."""
        # Initialize data loading pipeline (loads existing data only)
        pipeline = DataLoadingPipeline(
            batch_size=self.batch_size,
            config_path=self.config_path
        )

        # Run the pipeline to get dataloaders from existing data
        self.train_dataloader, self.val_dataloader = pipeline.run_pipeline()
        
        # Initialize validation samples for plotting
        self.initialize_validation_samples()
        
        # Calculate total batches for progress tracking
        self.total_batches = len(self.train_dataloader) * self.num_epochs
    
    def initialize_validation_samples(self):
        """Initialize validation samples for consistent plotting across epochs."""
        if self.val_dataloader is not None:
            # Get random validation samples for consistent plotting
            self.validation_sample_indices, states, next_states, actions = get_random_validation_samples(
                self.val_dataloader, 
                n_samples=self.autoencoder_config['num_validation_samples'],
                seed=self.autoencoder_config['validation_seed']
            )
            # For autoencoder, we use current states as both input and target
            self.validation_samples = (states, states, actions)  # Use states as both input and target
    
    def generate_validation_plots(self, epoch: int):
        """Generate combined validation plots for the current epoch."""
        if self.validation_samples is None:
            return
            
        states, target_states, actions = self.validation_samples
        
        # Move to device
        states = states.to(self.device)
        target_states = target_states.to(self.device)
        
        with torch.no_grad():
            # Get predictions for validation samples
            # states shape: [B, T, C, H, W] where T=4 (4 frames)
            z_state = self.encoder(states)  # [B, T, embed_dim] where T=4
            
            # For plotting, we'll use the first frame as representative
            # Extract first frame latent: [B, T, embed_dim] -> [B, 1, embed_dim]
            z_first_frame = z_state[:, 0:1, :]  # [B, 1, embed_dim]
            
            # Decode first frame latent
            reconstructed_first_frame = self.decoder(z_first_frame)  # [B, C, 1, H, W]
            
            # Convert from [B, C, 1, H, W] to [B, 1, C, H, W] for plotting
            reconstructed_states = reconstructed_first_frame.transpose(1, 2)
            
            # For plotting, use first frame of input states
            first_frame_states = states[:, 0, :, :, :].unsqueeze(1)  # [B, 1, C, H, W]
            first_frame_targets = target_states[:, 0, :, :, :].unsqueeze(1)  # [B, 1, C, H, W]
            
            # Generate combined plot with input state and reconstructed state
            plot_encoder_decoder_combined_validation_samples(
                current_states=first_frame_states,
                predicted_next_states=reconstructed_states,
                true_next_states=first_frame_targets,
                epoch=epoch,
                sample_indices=self.validation_sample_indices,
                output_dir=self.plot_output_dir,
                model_name="autoencoder"
            )
            
        # Plots saved silently
        pass
    
    def save_best_weights(self):
        """Save current model weights as best weights."""
        import copy
        self.best_weights = {
            'encoder': copy.deepcopy(self.encoder.state_dict()),
            'decoder': copy.deepcopy(self.decoder.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
        }
        if self.lr_scheduler is not None:
            self.best_weights['lr_scheduler'] = copy.deepcopy(self.lr_scheduler.state_dict())
    
    def restore_best_weights(self):
        """Restore the best model weights."""
        if self.best_weights is not None:
            self.encoder.load_state_dict(self.best_weights['encoder'])
            self.decoder.load_state_dict(self.best_weights['decoder'])
            self.optimizer.load_state_dict(self.best_weights['optimizer'])
            if self.lr_scheduler is not None and 'lr_scheduler' in self.best_weights:
                self.lr_scheduler.load_state_dict(self.best_weights['lr_scheduler'])
            # Best model weights restored silently
            pass
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early based on validation loss.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if not self.early_stopping_enabled or val_loss is None:
            return False
        
        # Check if validation loss improved (lower is better for loss)
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            # Improvement found
            improvement = self.best_val_loss - val_loss
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            # Save best weights
            if self.restore_best_weights:
                self.save_best_weights()
            return False
        else:
            # No improvement
            self.epochs_without_improvement += 1
            
            # Check if patience exceeded
            if self.epochs_without_improvement >= self.early_stopping_patience:
                return True
        
        return False
            
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single training step with autoencoder reconstruction.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Reconstruction loss for this batch
        """
        state, next_state, action, reward = batch

        # Move to device
        state = state.to(self.device)
        
        # Normalize images to [0, 1] range
        state = state / 255.0
        
        # Forward pass through encoder-decoder
        self.optimizer.zero_grad()
        # 1. Encode current state to latent representation
        # state shape: [B, T, C, H, W] where T=4 (4 frames)
        z_state = self.encoder(state)  # [B, T, embed_dim] where T=4
        # 2. Process each frame individually for autoencoder reconstruction
        total_reconstruction_loss = 0.0
        num_frames = z_state.shape[1]  # Should be 4
        
        for frame_idx in range(num_frames):
            # Extract single frame latent: [B, T, embed_dim] -> [B, 1, embed_dim]
            z_frame = z_state[:, frame_idx:frame_idx+1, :]  # [B, 1, embed_dim]
            
            # Decode single frame latent to reconstruct the same frame
            frame_reconstructed = self.decoder(z_frame)  # [B, C, 1, H, W]
            # Extract corresponding input frame: [B, T, C, H, W] -> [B, C, H, W]
            frame_input = state[:, frame_idx, :, :, :]  # [B, C, H, W]
            
            # Add temporal dimension to match decoder output: [B, C, H, W] -> [B, C, 1, H, W]
            frame_target = frame_input.unsqueeze(2)  # [B, C, 1, H, W]
            
            # Compute MSE reconstruction loss for this frame
            frame_loss = self.criterion(frame_reconstructed, frame_target)
            total_reconstruction_loss += frame_loss
        
        # Average loss across all frames
        reconstruction_loss = total_reconstruction_loss / num_frames
        
        # 3. Backward pass through encoder-decoder graph
        reconstruction_loss.backward()
        
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            # Include all trainable parameters in gradient clipping
            trainable_params = (list(self.encoder.parameters()) + 
                              list(self.decoder.parameters()))
            torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clipping)
        self.optimizer.step()
        
        return reconstruction_loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step with autoencoder reconstruction.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Reconstruction loss for this batch
        """
        state, next_state, action, reward = batch

        # Move to device
        state = state.to(self.device)
        
        # Normalize images to [0, 1] range
        state = state / 255.0
        
        with torch.no_grad():
            # 1. Encode current state
            # state shape: [B, T, C, H, W] where T=4 (4 frames)
            z_state = self.encoder(state)  # [B, T, embed_dim] where T=4
            
            # 2. Process each frame individually for autoencoder reconstruction
            total_reconstruction_loss = 0.0
            num_frames = z_state.shape[1]  # Should be 4
            
            for frame_idx in range(num_frames):
                # Extract single frame latent: [B, T, embed_dim] -> [B, 1, embed_dim]
                z_frame = z_state[:, frame_idx:frame_idx+1, :]  # [B, 1, embed_dim]
                
                # Decode single frame latent to reconstruct the same frame
                frame_reconstructed = self.decoder(z_frame)  # [B, C, 1, H, W]
                
                # Extract corresponding input frame: [B, T, C, H, W] -> [B, C, H, W]
                frame_input = state[:, frame_idx, :, :, :]  # [B, C, H, W]
                
                # Add temporal dimension to match decoder output: [B, C, H, W] -> [B, C, 1, H, W]
                frame_target = frame_input.unsqueeze(2)  # [B, C, 1, H, W]
                
                # Compute MSE reconstruction loss for this frame
                frame_loss = self.criterion(frame_reconstructed, frame_target)
                total_reconstruction_loss += frame_loss
            
            # Average loss across all frames
            reconstruction_loss = total_reconstruction_loss / num_frames
            
        return reconstruction_loss.item()
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average reconstruction loss for the epoch
        """
        self.encoder.train()
        self.decoder.train()
        
        total_reconstruction_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # Update current epoch counter
        self.current_epoch += 1
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            reconstruction_loss = self.train_step(batch)
            total_reconstruction_loss += reconstruction_loss
            
            # Update batch counter
            self.batches_completed += 1
            
            # Batch progress logged silently
            
            # Log batch loss to wandb (minimal terminal output)
            if wandb.run is not None:
                log_dict = {
                    "batch_reconstruction_loss": reconstruction_loss,
                    "batch": batch_idx,
                    "epoch": self.current_epoch,
                    "total_batches_completed": self.batches_completed,
                    "training_progress": self.batches_completed / self.total_batches
                }
                wandb.log(log_dict)
                
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        return avg_reconstruction_loss
        
    def validate_epoch(self) -> Optional[float]:
        """
        Validate for one epoch.
        
        Returns:
            Average reconstruction loss for the epoch, or None if no validation data
        """
        if self.val_dataloader is None:
            return None
            
        self.encoder.eval()
        self.decoder.eval()
        
        total_reconstruction_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch_idx, batch in enumerate(self.val_dataloader):
            reconstruction_loss = self.validate_step(batch)
            total_reconstruction_loss += reconstruction_loss
            
            # Validation batch progress logged silently
            
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        return avg_reconstruction_loss
        
    def save_checkpoint(self, epoch: int, train_reconstruction_loss: float,
                       val_reconstruction_loss: Optional[float]):
        """
        Save model checkpoints for all models.
        
        Args:
            epoch: Current epoch number
            train_reconstruction_loss: Training reconstruction loss for this epoch
            val_reconstruction_loss: Validation reconstruction loss for this epoch (if available)
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_reconstruction_loss': train_reconstruction_loss,
            'val_reconstruction_loss': val_reconstruction_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Use reconstruction loss for best model selection
        current_val_loss = val_reconstruction_loss if val_reconstruction_loss is not None else train_reconstruction_loss
        
        # Save best checkpoint if validation loss improved
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            
            # Also save individual model state dicts for easy loading
            torch.save(self.encoder.state_dict(), self.checkpoint_dir / "best_encoder.pth")
            torch.save(self.decoder.state_dict(), self.checkpoint_dir / "best_decoder.pth")
            
    def train(self):
        """Run the complete end-to-end training loop."""
        # Initialize everything
        self.initialize_models()
        self.initialize_optimizer()
        self.load_data()
        
        # Initialize wandb if configured
        if self.autoencoder_config['wandb_enabled']:
            wandb.init(
                project=self.autoencoder_config['wandb_project'],
                entity=self.autoencoder_config['wandb_entity'],
                name=f"autoencoder-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_reconstruction_loss = self.train_epoch()
            
            # Validate
            val_reconstruction_loss = self.validate_epoch()
            
            # Generate validation plots for this epoch
            self.generate_validation_plots(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Calculate and display progress
            progress_percentage = (self.batches_completed / self.total_batches) * 100
            remaining_batches = self.total_batches - self.batches_completed
            
            # Log to wandb
            log_dict = {
                "epoch": epoch,
                "current_epoch": self.current_epoch,
                "train_reconstruction_loss": train_reconstruction_loss,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "total_batches_completed": self.batches_completed,
                "training_progress": progress_percentage,
                "remaining_batches": remaining_batches
            }
            if val_reconstruction_loss is not None:
                log_dict["val_reconstruction_loss"] = val_reconstruction_loss
                
            if wandb.run is not None:
                wandb.log(log_dict)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_reconstruction_loss, val_reconstruction_loss)
            
            # Step learning rate scheduler
            if self.lr_scheduler is not None:
                # Use validation reconstruction loss for plateau scheduler, otherwise step normally
                step_scheduler(self.lr_scheduler, val_reconstruction_loss)

            # Check for early stopping (use validation reconstruction loss as primary objective)
            if val_reconstruction_loss is not None:
                if self.check_early_stopping(val_reconstruction_loss):
                    # Restore best weights if configured
                    if self.restore_best_weights and self.best_weights is not None:
                        self.restore_best_weights()
                    break

            # Terminal output suppressed for clean execution
            
        if wandb.run is not None:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder models end-to-end')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = AutoencoderTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
