#!/usr/bin/env python3
"""
Dynamics-Based Reward Predictor Training Script

This script implements supervised training of a reward predictor model using latent 
representations generated from a learned dynamics model. The goal is to estimate the 
reward using the predicted next latent state from the dynamics model, rather than the 
encoded ground-truth next state.

The script supports two training loops:
1. Training with JEPA-trained encoder/predictor models
2. Training with encoder-decoder-trained encoder/predictor models

Both loops follow the same structure with different model weight paths.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Set MPS fallback for unsupported operations and add project root to path
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, init_reward_predictor, load_config
from src.scripts.collect_load_data import DataCollectionPipeline
from src.utils.set_device import set_device


class DynamicsRewardPredictorTrainer:
    """
    Trainer class for supervised dynamics-based reward predictor training.
    
    This class handles the complete training pipeline for both JEPA and 
    encoder-decoder approaches including:
    - Model initialization (encoder, predictor, reward predictor)
    - Pre-trained weight loading
    - Data loading and preprocessing
    - Training and validation loops with dynamics prediction
    - Checkpointing and logging
    """
    
    def __init__(self, config_path: str = None, approach: str = "jepa"):
        """
        Initialize the dynamics reward predictor trainer.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            approach: Training approach - either "jepa" or "encoder_decoder"
        """
        if approach not in ["jepa", "encoder_decoder"]:
            raise ValueError("Approach must be either 'jepa' or 'encoder_decoder'")
            
        self.config_path = config_path
        self.approach = approach
        self.config = load_config(config_path)
        
        # Training parameters from config
        self.training_config = self.config['training']['dynamics_reward_predictor']
        self.num_epochs = self.training_config['num_epochs']
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = self.training_config['learning_rate']
        self.weight_decay = self.training_config['weight_decay']
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models
        self.encoder = None
        self.predictor = None
        self.reward_predictor = None
        
        # Training components
        self.optimizer = None
        self.criterion = nn.MSELoss()  # Mean Squared Error for reward prediction
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(f"weights/{self.approach}/dynamics_reward_predictor")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for pre-trained models
        self.pretrained_dir = Path(f"weights/{self.approach}")
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the trainer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_dynamics_reward_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self):
        """Initialize encoder, predictor, and reward predictor models."""
        self.logger.info(f"Initializing models for {self.approach} approach...")
        
        # Initialize models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)
        self.reward_predictor = init_reward_predictor(self.config_path).to(self.device)
        
        self.logger.info(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        self.logger.info(f"Predictor parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
        self.logger.info(f"Reward predictor parameters: {sum(p.numel() for p in self.reward_predictor.parameters()):,}")
        
    def load_pretrained_weights(self):
        """Load pre-trained encoder and predictor weights."""
        self.logger.info(f"Loading pre-trained weights from {self.pretrained_dir}...")
        
        # Load encoder weights
        encoder_path = self.pretrained_dir / "best_encoder.pth"
        if encoder_path.exists():
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.logger.info(f"Loaded encoder weights from {encoder_path}")
        else:
            raise FileNotFoundError(f"Pre-trained encoder not found at {encoder_path}")
            
        # Load predictor weights
        predictor_path = self.pretrained_dir / "best_predictor.pth"
        if predictor_path.exists():
            self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
            self.logger.info(f"Loaded predictor weights from {predictor_path}")
        else:
            raise FileNotFoundError(f"Pre-trained predictor not found at {predictor_path}")
            
        # Set encoder and predictor to evaluation mode and disable gradients
        self.encoder.eval()
        self.predictor.eval()
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        for param in self.predictor.parameters():
            param.requires_grad = False
            
        self.logger.info("Encoder and predictor set to evaluation mode with gradients disabled")
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer for reward predictor parameters only."""
        # Only optimize reward predictor parameters
        self.optimizer = optim.AdamW(
            self.reward_predictor.parameters(),
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
        Perform a single training step using dynamics prediction.
        
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

        # Take the last action value
        action = action[:, -1]
        action = action.to(self.device)

        # Take the last reward value 
        reward = reward[:, -1]
        reward = reward.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Encode current state with frozen encoder (no gradients)
        with torch.no_grad():
            z_state = self.encoder(state)          # [B, N_tokens, embed_dim]
            
            # Predict next latent state using the dynamics model (predictor)
            z_predicted_next_state = self.predictor(z_state, action)  # [B, N_tokens, embed_dim]
        
        # Predict reward using current latent state and predicted next latent state
        predicted_reward = self.reward_predictor(z_state, z_predicted_next_state)  # [B, 1, 1]
        
        # Reshape reward tensors for MSE loss
        # predicted_reward is [B, 1, 1], we need [B]
        predicted_reward = predicted_reward.squeeze(-1).squeeze(-1)  # [B]
        
        # Ensure reward is the right shape
        if reward.dim() > 1:
            reward = reward.squeeze()  # Remove extra dimensions if present
            
        # Compute MSE loss between predicted and ground-truth reward
        loss = self.criterion(predicted_reward, reward)
        
        # Backward pass (only reward predictor parameters will be updated)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step using dynamics prediction.
        
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

        # Take the last action value
        action = action[:, -1]
        action = action.to(self.device)

        # Take the last reward value
        reward = reward[:, -1]
        reward = reward.to(self.device)
        
        with torch.no_grad():
            # Encode current state
            z_state = self.encoder(state)
            
            # Predict next latent state using the dynamics model (predictor)
            z_predicted_next_state = self.predictor(z_state, action)
            
            # Predict reward using current latent state and predicted next latent state
            predicted_reward = self.reward_predictor(z_state, z_predicted_next_state)
            
            # Reshape reward tensors for MSE loss
            predicted_reward = predicted_reward.squeeze(-1).squeeze(-1)  # [B]
            
            # Ensure reward is the right shape
            if reward.dim() > 1:
                reward = reward.squeeze()
                
            # Compute loss
            loss = self.criterion(predicted_reward, reward)
            
        return loss.item()
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        # Set reward predictor to training mode (encoder/predictor remain in eval)
        self.reward_predictor.train()
        
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
            
        self.reward_predictor.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
            
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """
        Save model checkpoint if validation loss improved.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch (can be None)
        """
        # Use validation loss if available, otherwise use training loss
        current_loss = val_loss if val_loss is not None else train_loss
        
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            
            # Save reward predictor state dict
            checkpoint_path = self.checkpoint_dir / "best_dynamics_reward_predictor.pth"
            torch.save(self.reward_predictor.state_dict(), checkpoint_path)
            
            # Also save a complete checkpoint with training info
            full_checkpoint = {
                'epoch': epoch,
                'reward_predictor_state_dict': self.reward_predictor.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'approach': self.approach
            }
            
            full_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(full_checkpoint, full_checkpoint_path)
            
            loss_type = "validation" if val_loss is not None else "training"
            self.logger.info(f"New best {loss_type} loss: {current_loss:.6f} - saved checkpoint")
            
    def train(self):
        """Run the complete training loop."""
        self.logger.info(f"Starting dynamics reward predictor training with {self.approach} approach...")
        
        # Initialize everything
        self.initialize_models()
        self.load_pretrained_weights()
        self.initialize_optimizer()
        self.load_data()
        
        # Initialize wandb if configured
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'simple-rl-worlds'),
                entity=wandb_config.get('entity'),
                name=f"dynamics-reward-predictor-{self.approach}-{time.strftime('%Y%m%d-%H%M%S')}",
                config={**self.config, 'approach': self.approach},
                tags=["dynamics-reward-predictor", self.approach]
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
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "approach": self.approach
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
            
        self.logger.info(f"Dynamics reward predictor training with {self.approach} approach completed!")
        
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()


def train_from_config(config_path: str = None):
    """
    Train dynamics reward predictors based on the 'versions' parameter in config.yaml.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
    """
    from src.utils.init_models import load_config
    
    config = load_config(config_path)
    versions = config['training']['dynamics_reward_predictor'].get('versions', 'both')
    
    print(f"Training versions specified in config: {versions}")
    
    if versions == 'both':
        approaches = ["jepa", "encoder_decoder"]
    elif versions == 'jepa':
        approaches = ["jepa"]
    elif versions == 'encoder_decoder':
        approaches = ["encoder_decoder"]
    else:
        raise ValueError(f"Invalid versions parameter: {versions}. Must be 'both', 'jepa', or 'encoder_decoder'")
    
    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"Starting {approach.upper()} dynamics reward predictor training")
        print(f"{'='*60}")
        
        try:
            trainer = DynamicsRewardPredictorTrainer(config_path=config_path, approach=approach)
            trainer.train()
            print(f"\n{approach.upper()} training completed successfully!")
        except FileNotFoundError as e:
            print(f"\nSkipping {approach} training: {e}")
        except Exception as e:
            print(f"\nError during {approach} training: {e}")
            raise
        
        print(f"\n{approach.upper()} training finished.")


def train_both_approaches(config_path: str = None):
    """
    Train dynamics reward predictors for both JEPA and encoder-decoder approaches.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
    """
    approaches = ["jepa", "encoder_decoder"]
    
    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"Starting {approach.upper()} dynamics reward predictor training")
        print(f"{'='*60}")
        
        try:
            trainer = DynamicsRewardPredictorTrainer(config_path=config_path, approach=approach)
            trainer.train()
            print(f"\n{approach.upper()} training completed successfully!")
        except FileNotFoundError as e:
            print(f"\nSkipping {approach} training: {e}")
        except Exception as e:
            print(f"\nError during {approach} training: {e}")
            raise
        
        print(f"\n{approach.upper()} training finished.")


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train dynamics-based reward predictor models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    parser.add_argument('--approach', type=str, choices=['jepa', 'encoder_decoder', 'both', 'from_config'],
                       default='from_config',
                       help='Training approach: jepa, encoder_decoder, both, or from_config (reads versions from config.yaml)')
    
    args = parser.parse_args()
    
    if args.approach == 'from_config':
        # Use versions parameter from config
        train_from_config(config_path=args.config)
    elif args.approach == 'both':
        # Train both approaches (override config)
        train_both_approaches(config_path=args.config)
    else:
        # Train single approach (override config)
        trainer = DynamicsRewardPredictorTrainer(config_path=args.config, approach=args.approach)
        trainer.train()


if __name__ == "__main__":
    main()
