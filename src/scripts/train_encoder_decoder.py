#!/usr/bin/env python3
"""
Encoder-Decoder Training Script

This script implements end-to-end training of the complete model stack:
encoder, predictor, and decoder. The main objective is to train all components
jointly to reconstruct the next state frame from the current state and action.

The training process:
1. Encode current state to latent representation
2. Predict next latent state using current latent state and action
3. Decode predicted latent state to reconstruct next frame
4. Optimize L1 reconstruction loss through the entire model graph
"""

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

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, init_decoder, load_config, init_reward_predictor
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device
from src.utils.scheduler_utils import create_lr_scheduler, step_scheduler, get_current_lr


class EncoderDecoderTrainer:
    """
    Trainer class for end-to-end Encoder-Predictor-Decoder architecture.
    
    This class handles the complete training pipeline including:
    - Model initialization (encoder, predictor, decoder)
    - Data loading and preprocessing
    - Training and validation loops with reconstruction loss
    - Joint optimization of all model components
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Encoder-Decoder trainer.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Training parameters
        self.training_config = self.config['training']['main_loops']
        self.num_epochs = self.training_config['num_epochs']
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = self.training_config['learning_rate']
        self.weight_decay = self.training_config['weight_decay']
        self.gradient_clipping = self.training_config.get('gradient_clipping', None)
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models
        self.encoder = None
        self.predictor = None
        self.decoder = None
        self.reward_predictor = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = nn.L1Loss()  # Mean Absolute Error for reconstruction
        self.reward_criterion = nn.MSELoss()  # Reward prediction loss
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/encoder_decoder")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Flag for reward predictor usage
        self.use_reward_predictor = self.config['training'].get('main_loops').get('reward_loss')
        
    def initialize_models(self):
        """Initialize encoder, predictor, and decoder models."""
        # Initialize all models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)
        self.decoder = init_decoder(self.config_path).to(self.device)

        if self.use_reward_predictor:
            self.reward_predictor = init_reward_predictor(self.config_path).to(self.device)
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer and learning rate scheduler for all trainable parameters."""
        # Combine parameters from all models
        trainable_params = (list(self.encoder.parameters()) + 
                          list(self.predictor.parameters()) + 
                          list(self.decoder.parameters()))

        if self.use_reward_predictor:
            trainable_params += list(self.reward_predictor.parameters())
        
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
        # Initialize data loading pipeline (loads existing data only)
        pipeline = DataLoadingPipeline(
            batch_size=self.batch_size,
            config_path=self.config_path
        )

        # Run the pipeline to get dataloaders from existing data
        self.train_dataloader, self.val_dataloader = pipeline.run_pipeline()
        
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float, float]:
        """
        Perform a single training step with end-to-end reconstruction.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Tuple of (reconstruction_loss, reward_loss, total_loss) for this batch
        """
        state, next_state, action, reward = batch

        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward[:, -1]
        reward = reward.to(self.device)
        
        # Forward pass through the entire model stack
        self.optimizer.zero_grad()
        
        # 1. Encode current state to latent representation
        z_state = self.encoder(state)  # [B, N_tokens, embed_dim]
        
        # 2. Predict next latent state using current latent state and action sequence
        # Pass the full action sequence to the predictor (now supports [B, T] actions)
        z_next_pred = self.predictor(z_state, action)
        
        # 3. Decode predicted latent state to reconstruct next frame
        next_state_reconstructed = self.decoder(z_next_pred)
        
        # 4. Compute L1 reconstruction loss between reconstructed and ground-truth next state
        reconstruction_loss = self.criterion(next_state_reconstructed, next_state)
        
        # Initialize total loss with reconstruction loss
        total_loss = reconstruction_loss
        reward_loss = 0.0
        
        # 5. Add reward prediction loss if enabled
        if self.use_reward_predictor:
            # Encode ground-truth next state for reward prediction
            z_next_target = self.encoder(next_state)
            
            # Predict reward using current and ground-truth next latent states
            reward_pred = self.reward_predictor(z_state, z_next_target)
            reward_pred = reward_pred.squeeze(-1).squeeze(-1)  # Ensure shape is [B]
            
            # Compute reward loss (MSE loss)
            reward_loss = self.reward_criterion(reward_pred, reward)
            total_loss += reward_loss
        
        # 6. Backward pass through entire model graph
        total_loss.backward()
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            # Include all trainable parameters in gradient clipping
            trainable_params = (list(self.encoder.parameters()) + 
                              list(self.predictor.parameters()) + 
                              list(self.decoder.parameters()))
            if self.use_reward_predictor:
                trainable_params += list(self.reward_predictor.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clipping)
        self.optimizer.step()
        
        # Convert tensor losses to float for consistent return type
        if isinstance(reward_loss, torch.Tensor):
            reward_loss = reward_loss.item()
        
        return reconstruction_loss.item(), reward_loss, total_loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float, float]:
        """
        Perform a single validation step with end-to-end reconstruction.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Tuple of (reconstruction_loss, reward_loss, total_loss) for this batch
        """
        state, next_state, action, reward = batch

        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward[:, -1]
        reward = reward.to(self.device)
        
        with torch.no_grad():
            # 1. Encode current state
            z_state = self.encoder(state)
            
            # 2. Predict next latent state using full action sequence
            z_next_pred = self.predictor(z_state, action)
            
            # 3. Decode predicted latent state
            next_state_reconstructed = self.decoder(z_next_pred)
            
            # 4. Compute reconstruction loss
            reconstruction_loss = self.criterion(next_state_reconstructed, next_state)
            
            # Initialize total loss with reconstruction loss
            total_loss = reconstruction_loss
            reward_loss = 0.0
            
            # 5. Compute reward prediction loss if enabled
            if self.use_reward_predictor:
                # Encode ground-truth next state for reward prediction
                z_next_target = self.encoder(next_state)
                
                # Predict reward using current and ground-truth next latent states
                reward_pred = self.reward_predictor(z_state, z_next_target)
                reward_pred = reward_pred.squeeze(-1).squeeze(-1)  # Ensure shape is [B]
                
                # Compute reward loss (MSE loss)
                reward_loss = self.reward_criterion(reward_pred, reward)
                total_loss += reward_loss
            
            # Convert tensor losses to float for consistent return type
            if isinstance(reward_loss, torch.Tensor):
                reward_loss = reward_loss.item()
            
        return reconstruction_loss.item(), reward_loss, total_loss.item()
        
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_reconstruction_loss, average_reward_loss, average_total_loss) for the epoch
        """
        self.encoder.train()
        self.predictor.train()
        self.decoder.train()
        if self.use_reward_predictor:
            self.reward_predictor.train()
        
        total_reconstruction_loss = 0.0
        total_reward_loss = 0.0
        total_total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            reconstruction_loss, reward_loss, batch_total_loss = self.train_step(batch)
            total_reconstruction_loss += reconstruction_loss
            total_reward_loss += reward_loss
            total_total_loss += batch_total_loss
            
            # Log batch loss to wandb (minimal terminal output)
            if wandb.run is not None:
                log_dict = {
                    "batch_reconstruction_loss": reconstruction_loss,
                    "batch_total_loss": batch_total_loss,
                    "batch": batch_idx
                }
                if self.use_reward_predictor:
                    log_dict["batch_reward_loss"] = reward_loss
                wandb.log(log_dict)
                
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_total_loss = total_total_loss / num_batches
        return avg_reconstruction_loss, avg_reward_loss, avg_total_loss
        
    def validate_epoch(self) -> Optional[Tuple[float, float, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_reconstruction_loss, average_reward_loss, average_total_loss) for the epoch, or None if no validation data
        """
        if self.val_dataloader is None:
            return None
            
        self.encoder.eval()
        self.predictor.eval()
        self.decoder.eval()
        if self.use_reward_predictor:
            self.reward_predictor.eval()
        
        total_reconstruction_loss = 0.0
        total_reward_loss = 0.0
        total_total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            reconstruction_loss, reward_loss, batch_total_loss = self.validate_step(batch)
            total_reconstruction_loss += reconstruction_loss
            total_reward_loss += reward_loss
            total_total_loss += batch_total_loss
            
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_total_loss = total_total_loss / num_batches
        return avg_reconstruction_loss, avg_reward_loss, avg_total_loss
        
    def save_checkpoint(self, epoch: int, train_reconstruction_loss: float, train_reward_loss: float, train_total_loss: float,
                       val_reconstruction_loss: Optional[float], val_reward_loss: Optional[float], val_total_loss: Optional[float]):
        """
        Save model checkpoints for all models.
        
        Args:
            epoch: Current epoch number
            train_reconstruction_loss: Training reconstruction loss for this epoch
            train_reward_loss: Training reward loss for this epoch
            train_total_loss: Training total loss for this epoch
            val_reconstruction_loss: Validation reconstruction loss for this epoch (if available)
            val_reward_loss: Validation reward loss for this epoch (if available)
            val_total_loss: Validation total loss for this epoch (if available)
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_reconstruction_loss': train_reconstruction_loss,
            'train_reward_loss': train_reward_loss,
            'train_total_loss': train_total_loss,
            'val_reconstruction_loss': val_reconstruction_loss,
            'val_reward_loss': val_reward_loss,
            'val_total_loss': val_total_loss,
            'config': self.config
        }
        
        # Add optional reward predictor state dict if it exists
        if self.reward_predictor is not None:
            checkpoint['reward_predictor_state_dict'] = self.reward_predictor.state_dict()
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Use reconstruction loss for best model selection (primary objective)
        current_val_loss = val_reconstruction_loss if val_reconstruction_loss is not None else train_reconstruction_loss
        
        # Save best checkpoint if validation loss improved
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            
            # Also save individual model state dicts for easy loading
            torch.save(self.encoder.state_dict(), self.checkpoint_dir / "best_encoder.pth")
            torch.save(self.predictor.state_dict(), self.checkpoint_dir / "best_predictor.pth")
            torch.save(self.decoder.state_dict(), self.checkpoint_dir / "best_decoder.pth")
            
            # Save reward predictor if it exists
            if self.reward_predictor is not None:
                torch.save(self.reward_predictor.state_dict(), self.checkpoint_dir / "best_reward_predictor.pth")
            
    def train(self):
        """Run the complete end-to-end training loop."""
        # Initialize everything
        self.initialize_models()
        self.initialize_optimizer()
        self.load_data()
        
        # Initialize wandb if configured
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'encoder-decoder-training'),
                entity=wandb_config.get('entity'),
                name=f"encoder-decoder-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_reconstruction_loss, train_reward_loss, train_total_loss = self.train_epoch()
            
            # Validate
            val_results = self.validate_epoch()
            if val_results is not None:
                val_reconstruction_loss, val_reward_loss, val_total_loss = val_results
            else:
                val_reconstruction_loss, val_reward_loss, val_total_loss = None, None, None
            
            epoch_time = time.time() - epoch_start_time
            
            # Log to wandb
            log_dict = {
                "epoch": epoch,
                "train_reconstruction_loss": train_reconstruction_loss,
                "train_total_loss": train_total_loss,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            if val_reconstruction_loss is not None:
                log_dict["val_reconstruction_loss"] = val_reconstruction_loss
                log_dict["val_total_loss"] = val_total_loss

            if self.use_reward_predictor:
                log_dict.update({
                    "train_reward_loss": train_reward_loss,
                })
                if val_reward_loss is not None:
                    log_dict["val_reward_loss"] = val_reward_loss
                
            if wandb.run is not None:
                wandb.log(log_dict)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_reconstruction_loss, train_reward_loss, train_total_loss,
                               val_reconstruction_loss, val_reward_loss, val_total_loss)
            
            # Step learning rate scheduler
            if self.lr_scheduler is not None:
                # Use validation reconstruction loss for plateau scheduler, otherwise step normally
                step_scheduler(self.lr_scheduler, val_reconstruction_loss)

            # Terminal output
            if val_reconstruction_loss is not None:
                # Base message with reconstruction and total losses
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Recon: {train_reconstruction_loss:.6f}, Val Recon: {val_reconstruction_loss:.6f}, Train Total: {train_total_loss:.6f}, Val Total: {val_total_loss:.6f}"
                
                # Add reward losses if enabled
                if self.use_reward_predictor:
                    message += f", Train Reward: {train_reward_loss:.6f}, Val Reward: {val_reward_loss:.6f}"
                
                print(message)
            else:
                # Base message for no validation with reconstruction and total losses
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Recon: {train_reconstruction_loss:.6f}, Val Recon: N/A, Train Total: {train_total_loss:.6f}, Val Total: N/A"
                
                # Add reward loss if enabled
                if self.use_reward_predictor:
                    message += f", Train Reward: {train_reward_loss:.6f}, Val Reward: N/A"
                
                print(message)
            
        if wandb.run is not None:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Encoder-Decoder models end-to-end')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = EncoderDecoderTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
