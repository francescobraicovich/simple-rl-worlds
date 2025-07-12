#!/usr/bin/env python3
"""
JEPA Training Script

This script implements self-supervised training of encoder and predictor models using
Joint-Embedding Predictive Architecture (JEPA). The main objective is to train:
1. An encoder to produce powerful latent representations of states
2. A predictor to model dynamics in the latent space

The training uses a target encoder (EMA of main encoder) for stable targets.
"""

import sys
import time
import copy
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

from src.utils.init_models import init_encoder, init_predictor, load_config
from src.scripts.collect_load_data import DataCollectionPipeline


class JEPATrainer:
    """
    Trainer class for Joint-Embedding Predictive Architecture (JEPA).
    
    This class handles the complete training pipeline including:
    - Model initialization (encoder, predictor, target encoder)
    - Data loading and preprocessing
    - Training and validation loops
    - Exponential Moving Average (EMA) updates for target encoder
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the JEPA trainer.
        
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
        self.ema_decay = self.training_config['ema_decay']
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.encoder = None
        self.predictor = None
        self.target_encoder = None
        
        # Training components
        self.optimizer = None
        self.criterion = nn.L1Loss()  # Mean Absolute Error
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/jepa")
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
        """Initialize encoder, predictor, and target encoder models."""
        self.logger.info("Initializing models...")
        
        # Initialize main models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)
        
        # Initialize target encoder as a copy of the main encoder
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        
        # Freeze target encoder (no gradients)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.logger.info(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        self.logger.info(f"Predictor parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer for trainable parameters."""
        # Combine parameters from encoder and predictor
        trainable_params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        
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
                
            
    def update_target_encoder(self):
        """Update target encoder weights using Exponential Moving Average (EMA)."""
        with torch.no_grad():
            for target_param, encoder_param in zip(self.target_encoder.parameters(), 
                                                 self.encoder.parameters()):
                target_param.data = (self.ema_decay * target_param.data + 
                                   (1 - self.ema_decay) * encoder_param.data)
                
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Loss value for this batch
        """
        state, next_state, action, reward = batch
        
        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Encode current state with main encoder
        print(f'State shape: {state.shape}, Next state shape: {next_state.shape}, Action shape: {action.shape}')
        z_state = self.encoder(state)  # [B, N_tokens, embed_dim]
        print(f'Encoded state shape: {z_state.shape}')
        
        # For JEPA, we need the action from the sequence
        # The dataset returns action sequences, we'll use the last action
        if action.dim() > 1:
            action_for_prediction = action[:, -1]  # Use last action in sequence
        else:
            action_for_prediction = action
            
        # Predict next latent state
        z_next_pred = self.predictor(z_state, action_for_prediction)
        print(f'Predicted next state shape: {z_next_pred.shape}')
        
        # Encode ground-truth next state with target encoder (no gradients)
        with torch.no_grad():
            z_next_target = self.target_encoder(next_state)
            print(f'Ground-truth next state shape: {z_next_target.shape}')

        # Compute L1 loss between predicted and target latent states
        loss = self.criterion(z_next_pred, z_next_target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update target encoder with EMA
        self.update_target_encoder()
        
        return loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Loss value for this batch
        """
        state, next_state, action, reward = batch
        
        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        
        with torch.no_grad():
            # Encode current state
            z_state = self.encoder(state)
            
            # For JEPA, we need the action from the sequence
            if action.dim() > 1:
                action_for_prediction = action[:, -1]  # Use last action in sequence
            else:
                action_for_prediction = action
                
            # Predict next latent state
            z_next_pred = self.predictor(z_state, action_for_prediction)
            
            # Encode ground-truth next state with target encoder
            z_next_target = self.target_encoder(next_state)
            
            # Compute loss
            loss = self.criterion(z_next_pred, z_next_target)
            
        return loss.item()
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.encoder.train()
        self.predictor.train()
        self.target_encoder.eval()  # Target encoder is always in eval mode
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Log batch loss to wandb (minimal terminal output)
            if wandb.run is not None:
                wandb.log({"batch_loss": loss, "batch": batch_idx})
                
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
            
        self.encoder.eval()
        self.predictor.eval()
        self.target_encoder.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
            
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """
        Save model checkpoints.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch (if available)
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'target_encoder_state_dict': self.target_encoder.state_dict(),
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
            
            self.logger.info(f"New best validation loss: {val_loss:.6f} - saved checkpoint")
            
    def train(self):
        """Run the complete training loop."""
        self.logger.info("Starting JEPA training...")
        
        # Initialize everything
        self.initialize_models()
        self.initialize_optimizer()

        # Find the batch size from the config
        self.load_data()
        
        # Initialize wandb if configured
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'jepa-training'),
                entity=wandb_config.get('entity'),
                name=f"jepa-{time.strftime('%Y%m%d-%H%M%S')}",
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
                "train_loss": train_loss,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
                
            if wandb.run is not None:
                wandb.log(log_dict)
            
            # Terminal output (minimal)
            if val_loss is not None:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                               f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                               f"Train Loss: {train_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss)
            
        self.logger.info("Training completed!")
        
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train JEPA models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = JEPATrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
