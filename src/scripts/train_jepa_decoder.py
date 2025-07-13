#!/usr/bin/env python3
"""
JEPA Decoder Training Script

This script trains only the decoder model, leveraging pre-trained representations
learned by the JEPA stage. The encoder and predictor models are loaded with
pre-trained weights and their parameters are frozen during training.

The training process:
1. Load pre-trained and frozen encoder and predictor.
2. Encode current state to latent representation (no_grad).
3. Predict next latent state using current latent state and action (no_grad).
4. Decode predicted latent state to reconstruct next frame.
5. Optimize L1 reconstruction loss only for the decoder.
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

from src.utils.init_models import init_encoder, init_predictor, init_decoder, load_config
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device


class JEPADecoderTrainer:
    """
    Trainer class for training only the Decoder component of a JEPA-like architecture.
    
    This class handles the complete training pipeline including:
    - Model initialization (loading pre-trained encoder/predictor, initializing decoder)
    - Freezing encoder and predictor weights
    - Data loading and preprocessing
    - Training and validation loops with reconstruction loss
    - Optimization of only the decoder parameters
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the JEPA Decoder trainer.
        
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
        
        # Training components
        self.optimizer = None
        self.criterion = nn.L1Loss()  # Mean Absolute Error for reconstruction
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/jepa_decoder")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_models(self):
        """
        Initialize encoder, predictor, and decoder models.
        Load pre-trained weights for encoder and predictor and freeze them.
        """
        # Initialize all models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)
        self.decoder = init_decoder(self.config_path).to(self.device)
        
        # Load pre-trained weights for encoder and predictor
        jepa_weights_dir = Path("weights/jepa")
        encoder_path = jepa_weights_dir / "best_encoder.pth"
        predictor_path = jepa_weights_dir / "best_predictor.pth"
        
        if encoder_path.exists():
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        else:
            print(f"Pre-trained encoder not found at {encoder_path}. Training from scratch.")
            
        if predictor_path.exists():
            self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        else:
            print(f"Pre-trained predictor not found at {predictor_path}. Training from scratch.")
            
        # Freeze encoder and predictor weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval() # Set to eval mode to disable dropout/batchnorm updates
        
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.predictor.eval() # Set to eval mode 
        
    def initialize_optimizer(self):
        """
        Initialize the AdamW optimizer only for the decoder's trainable parameters.
        """
        # Only pass decoder parameters to the optimizer
        self.optimizer = optim.AdamW(
            self.decoder.parameters(), # Only decoder parameters
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
    def load_data(self):
        """
        Load training and validation data using DataLoadingPipeline.
        """
        # Initialize data loading pipeline (loads existing data only)
        pipeline = DataLoadingPipeline(
            batch_size=self.batch_size,
            config_path=self.config_path
        )

        # Run the pipeline to get dataloaders from existing data
        self.train_dataloader, self.val_dataloader = pipeline.run_pipeline()
        
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single training step for the decoder.
        Encoder and predictor are used in no_grad context.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Reconstruction loss value for this batch
        """
        state, next_state, action, reward = batch

        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            # 1. Encode current state to latent representation
            z_state = self.encoder(state)  # [B, N_tokens, embed_dim]
            
            # 2. Predict next latent state using current latent state and action
            # The dataset returns action sequences, we'll use the last action
            if action.dim() > 1:
                action_for_prediction = action[:, -1]  # Use last action in sequence
            else:
                action_for_prediction = action
                
            z_next_pred = self.predictor(z_state.detach(), action_for_prediction) # Detach z_state

        # 3. Decode predicted latent state to reconstruct next frame
        next_state_reconstructed = self.decoder(z_next_pred.detach()) # Detach z_next_pred
        
        # 4. Compute L1 reconstruction loss between reconstructed and ground-truth next state
        loss = self.criterion(next_state_reconstructed, next_state)
        
        # 5. Backward pass (only decoder weights will receive gradients)
        loss.backward()
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(),
                self.gradient_clipping
            )
        self.optimizer.step()
        
        return loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step for the decoder.
        Encoder and predictor are used in no_grad context.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Reconstruction loss value for this batch
        """
        state, next_state, action, reward = batch

        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        
        with torch.no_grad():
            # 1. Encode current state
            z_state = self.encoder(state)
            
            # 2. Predict next latent state
            if action.dim() > 1:
                action_for_prediction = action[:, -1]  # Use last action in sequence
            else:
                action_for_prediction = action
                
            z_next_pred = self.predictor(z_state, action_for_prediction)
            
            # 3. Decode predicted latent state
            next_state_reconstructed = self.decoder(z_next_pred)
            
            # 4. Compute reconstruction loss
            loss = self.criterion(next_state_reconstructed, next_state)
            
        return loss.item()
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training reconstruction loss for the epoch
        """
        # Encoder and predictor remain in eval mode (frozen)
        self.decoder.train()
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Log batch loss to wandb (minimal terminal output)
            if wandb.run is not None:
                wandb.log({"batch_reconstruction_loss": loss, "batch": batch_idx})
                
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def validate_epoch(self) -> Optional[float]:
        """
        Validate for one epoch.
        
        Returns:
            Average validation reconstruction loss for the epoch, or None if no validation data
        """
        if self.val_dataloader is None:
            return None
            
        # Encoder and predictor remain in eval mode (frozen)
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
            
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """
        Save model checkpoints for the decoder.
        
        Args:
            epoch: Current epoch number
            train_loss: Training reconstruction loss for this epoch
            val_loss: Validation reconstruction loss for this epoch (if available)
        """
        checkpoint = {
            'epoch': epoch,
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_decoder_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if validation loss improved
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_decoder_checkpoint.pth"
            torch.save(checkpoint, best_path)
            
            # Also save individual model state dict for easy loading
            torch.save(self.decoder.state_dict(), self.checkpoint_dir / "best_decoder.pth")
            
    def train(self):
        """
        Run the complete decoder-only training loop.
        """
        # Initialize everything
        self.initialize_models()
        self.initialize_optimizer()
        self.load_data()
        
        # Initialize wandb if configured
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'jepa-decoder-training'),
                entity=wandb_config.get('entity'),
                name=f"jepa-decoder-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            epoch_time = time.time() - epoch_start_time
            
            # Log to wandb
            log_dict = {
                "epoch": epoch,
                "train_reconstruction_loss": train_loss,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            if val_loss is not None:
                log_dict["val_reconstruction_loss"] = val_loss
                
            if wandb.run is not None:
                wandb.log(log_dict)
                
            # Terminal output (minimal)
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: N/A")
                
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss)
            
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train JEPA Decoder model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = JEPADecoderTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
