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
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, init_decoder, load_config
from src.scripts.collect_load_data import DataCollectionPipeline
from src.utils.set_device import set_device


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
        self.checkpoint_dir = Path("weights/encoder_decoder")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the trainer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self):
        """Initialize encoder, predictor, and decoder models."""
        self.logger.info("Initializing models...")
        
        # Initialize all models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)
        self.decoder = init_decoder(self.config_path).to(self.device)
        
        self.logger.info(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        self.logger.info(f"Predictor parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
        self.logger.info(f"Decoder parameters: {sum(p.numel() for p in self.decoder.parameters()):,}")
        
        total_params = (sum(p.numel() for p in self.encoder.parameters()) +
                       sum(p.numel() for p in self.predictor.parameters()) +
                       sum(p.numel() for p in self.decoder.parameters()))
        self.logger.info(f"Total parameters: {total_params:,}")
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer for all trainable parameters."""
        # Combine parameters from all three models
        trainable_params = (list(self.encoder.parameters()) + 
                          list(self.predictor.parameters()) + 
                          list(self.decoder.parameters()))
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.logger.info(f"Optimizer initialized with lr={self.learning_rate}, weight_decay={self.weight_decay}")
        
    def load_data(self):
        """Load training and validation data using DataCollectionPipeline."""
        self.logger.info("Loading data...")
        
        # Initialize data collection pipeline
        pipeline = DataCollectionPipeline(
            batch_size=self.batch_size,
            config_path=self.config_path
        )

        # Run the full pipeline to get dataloaders
        self.train_dataloader, self.val_dataloader = pipeline.run_full_pipeline()
        
        # Update batch size in dataloaders if needed
        if self.train_dataloader.batch_size != self.batch_size:
            self.logger.warning(f"Dataloader batch size ({self.train_dataloader.batch_size}) "
                                f"differs from config batch size ({self.batch_size})")
            
        self.logger.info(f"Train batches: {len(self.train_dataloader)}")
        if self.val_dataloader:
            self.logger.info(f"Validation batches: {len(self.val_dataloader)}")
        else:
            self.logger.info("No validation data available")
                
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single training step with end-to-end reconstruction.
        
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
        
        # Forward pass through the entire model stack
        self.optimizer.zero_grad()
        
        # 1. Encode current state to latent representation
        z_state = self.encoder(state)  # [B, N_tokens, embed_dim]
        
        # 2. Predict next latent state using current latent state and action
        # The dataset returns action sequences, we'll use the last action
        if action.dim() > 1:
            action_for_prediction = action[:, -1]  # Use last action in sequence
        else:
            action_for_prediction = action
            
        z_next_pred = self.predictor(z_state, action_for_prediction)
        
        # 3. Decode predicted latent state to reconstruct next frame
        next_state_reconstructed = self.decoder(z_next_pred)
        
        # 4. Compute L1 reconstruction loss between reconstructed and ground-truth next state
        loss = self.criterion(next_state_reconstructed, next_state)
        
        # 5. Backward pass through entire model graph
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step with end-to-end reconstruction.
        
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
        self.encoder.train()
        self.predictor.train()
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
            
        self.encoder.eval()
        self.predictor.eval()
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
        Save model checkpoints for all three models.
        
        Args:
            epoch: Current epoch number
            train_loss: Training reconstruction loss for this epoch
            val_loss: Validation reconstruction loss for this epoch (if available)
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if validation loss improved
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            
            # Also save individual model state dicts for easy loading
            torch.save(self.encoder.state_dict(), self.checkpoint_dir / "best_encoder.pth")
            torch.save(self.predictor.state_dict(), self.checkpoint_dir / "best_predictor.pth")
            torch.save(self.decoder.state_dict(), self.checkpoint_dir / "best_decoder.pth")
            
            self.logger.info(f"New best validation loss: {val_loss:.6f} - saved checkpoint")
            
    def train(self):
        """Run the complete end-to-end training loop."""
        self.logger.info("Starting Encoder-Decoder training...")
        
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
            self.logger.info("Wandb initialized")
        else:
            self.logger.info("Wandb not configured or disabled")
        
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
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                               f"Train Reconstruction Loss: {train_loss:.6f}, "
                               f"Val Reconstruction Loss: {val_loss:.6f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                               f"Train Reconstruction Loss: {train_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss)
            
        self.logger.info("End-to-end training completed!")
        
        # Close wandb run
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
