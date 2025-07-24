#!/usr/bin/env python3
"""
JEPA Training Script

This script implements self-supervised training of encoder and predictor models using
Joint-Embedding Predictive Architecture (JEPA). The main objective is to train:
1. An encoder to produce powerful latent representations of states
2. A predictor to model dynamics in the latent space

The training uses a target encoder (EMA of main encoder) for stable targets.
"""

import os
# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import time
import copy
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, load_config, init_vicreg, init_reward_predictor
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device


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
        self.gradient_clipping = self.training_config.get('gradient_clipping', None)
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models
        self.encoder = None
        self.predictor = None
        self.target_encoder = None
        self.vicreg_loss = None
        self.reward_predictor = None
        
        # Training components
        self.optimizer = None
        self.criterion = nn.L1Loss()  # Mean Absolute Error
        self.reward_criterion = nn.MSELoss()  # Reward prediction loss

        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/jepa")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Flag for VICREG and reward predictor usage
        self.use_vicreg = self.config['models'].get('vicreg').get('active')
        self.use_reward_predictor = self.config['training'].get('main_loops').get('reward_loss')
        
    def initialize_models(self):
        """Initialize encoder, predictor, and target encoder models."""
        # Initialize main models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)

        if self.use_vicreg:
            self.vicreg_loss = init_vicreg(self.config_path).to(self.device)

        if self.use_reward_predictor:
            self.reward_predictor = init_reward_predictor(self.config_path).to(self.device)

        # Initialize target encoder as a copy of the main encoder
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        
        # Freeze target encoder (no gradients)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer for trainable parameters."""
        # Combine parameters from encoder and predictor
        trainable_params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.use_vicreg:
            trainable_params += list(self.vicreg_loss.parameters())

        if self.use_reward_predictor:
            trainable_params += list(self.reward_predictor.parameters())
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
    def load_data(self):
        """Load training and validation data using DataLoadingPipeline."""
        # Initialize data loading pipeline (loads existing data only)
        pipeline = DataLoadingPipeline(
            batch_size=self.batch_size,
            config_path=self.config_path
        )

        # Run the pipeline to get dataloaders from existing data
        self.train_dataloader, self.val_dataloader = pipeline.run_pipeline()
            
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
        reward = reward[:, -1]
        reward = reward.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()

        # Encode current state with main encoder
        z_state = self.encoder(state)  # [B, N_tokens, embed_dim]
        
        # Predict next latent state using the full action sequence
        z_next_pred = self.predictor(z_state, action)
        
        # Encode ground-truth next state
        if self.use_vicreg:
            # Use online encoder for VICReg (no EMA)
            z_next_target = self.encoder(next_state)
        else:
            # Use target encoder with EMA for standard training (no gradients)
            with torch.no_grad():
                z_next_target = self.target_encoder(next_state)

        # Initialize losses
        sim_loss, std_loss, cov_loss, reward_loss = 0.0, 0.0, 0.0, 0.0

        # Compute L1 loss between predicted and target latent states
        if self.use_vicreg:
            # Use VICReg loss if configured
            loss, sim_loss, std_loss, cov_loss = self.vicreg_loss(z_next_pred, z_next_target)
        else:
            # Default L1 loss
            loss = self.criterion(z_next_pred, z_next_target)

        if self.use_reward_predictor:
            # Predict reward using the reward predictor
            reward_pred = self.reward_predictor(z_state, z_next_target)
            reward_pred = reward_pred.squeeze(-1).squeeze(-1)  # Ensure shape is [B]
            # Compute reward loss (MSE loss)
            reward_loss = self.reward_criterion(reward_pred, reward)
            loss += reward_loss
    
        
        # Backward pass
        loss.backward()
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            # Include all trainable parameters in gradient clipping
            trainable_params = list(self.encoder.parameters()) + list(self.predictor.parameters())
            if self.use_vicreg:
                trainable_params += list(self.vicreg_loss.parameters())
            if self.use_reward_predictor:
                trainable_params += list(self.reward_predictor.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clipping)
        self.optimizer.step()
        
        # Update target encoder with EMA only if not using VICReg
        if not self.use_vicreg:
            self.update_target_encoder()
        
        # Convert tensor losses to float for consistent return type
        if isinstance(sim_loss, torch.Tensor):
            sim_loss = sim_loss.item()
        if isinstance(std_loss, torch.Tensor):
            std_loss = std_loss.item()
        if isinstance(cov_loss, torch.Tensor):
            cov_loss = cov_loss.item()
        if isinstance(reward_loss, torch.Tensor):
            reward_loss = reward_loss.item()
        
        return loss.item(), sim_loss, std_loss, cov_loss, reward_loss
        
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
        reward = reward[:, -1]
        reward = reward.to(self.device)

        with torch.no_grad():
            # Encode current state
            z_state = self.encoder(state)
            
            # Predict next latent state using full action sequence
            z_next_pred = self.predictor(z_state, action)
            
            # Encode ground-truth next state
            if self.use_vicreg:
                # Use online encoder for VICReg (no EMA)
                z_next_target = self.encoder(next_state)
            else:
                # Use target encoder with EMA for standard training
                z_next_target = self.target_encoder(next_state)

            sim_loss, std_loss, cov_loss, reward_loss = 0.0, 0.0, 0.0, 0.0

            # Compute loss
            if self.use_vicreg:
                # Use VICReg loss if configured
                loss, sim_loss, std_loss, cov_loss = self.vicreg_loss(z_next_pred, z_next_target)
            else:
                # Default L1 loss
                loss = self.criterion(z_next_pred, z_next_target)

            # If using reward predictor, compute reward loss
            if self.use_reward_predictor:
                # Predict reward using the reward predictor
                reward_pred = self.reward_predictor(z_state, z_next_target)
                reward_pred = reward_pred.squeeze(-1).squeeze(-1)  # Ensure shape is [B]
                # Compute reward loss (MSE loss)
                reward_loss = self.reward_criterion(reward_pred, reward)
                loss += reward_loss

            # Convert tensor losses to float for consistent return type
            if isinstance(sim_loss, torch.Tensor):
                sim_loss = sim_loss.item()
            if isinstance(std_loss, torch.Tensor):
                std_loss = std_loss.item()
            if isinstance(cov_loss, torch.Tensor):
                cov_loss = cov_loss.item()
            if isinstance(reward_loss, torch.Tensor):
                reward_loss = reward_loss.item()

        return loss.item(), sim_loss, std_loss, cov_loss, reward_loss

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
        total_sim_loss = 0.0
        total_std_loss = 0.0
        total_cov_loss = 0.0
        total_reward_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, sim_loss, std_loss, cov_loss, reward_loss = self.train_step(batch)
            total_loss += loss
            total_sim_loss += sim_loss
            total_std_loss += std_loss
            total_cov_loss += cov_loss
            total_reward_loss += reward_loss
            
            # Log batch loss to wandb (minimal terminal output)
            if wandb.run is not None:
                wandb.log({"batch_loss": loss, "batch": batch_idx,
                            "sim_loss": sim_loss, "std_loss": std_loss, 
                            "cov_loss": cov_loss, "reward_loss": reward_loss})

        avg_loss = total_loss / num_batches
        avg_sim_loss = total_sim_loss / num_batches
        avg_std_loss = total_std_loss / num_batches
        avg_cov_loss = total_cov_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        return avg_loss, avg_sim_loss, avg_std_loss, avg_cov_loss, avg_reward_loss
        
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
        total_sim_loss = 0.0
        total_std_loss = 0.0
        total_cov_loss = 0.0
        total_reward_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            loss, sim_loss, std_loss, cov_loss, reward_loss = self.validate_step(batch)
            total_loss += loss
            total_sim_loss += sim_loss
            total_std_loss += std_loss
            total_cov_loss += cov_loss
            total_reward_loss += reward_loss

        avg_loss = total_loss / num_batches
        avg_sim_loss = total_sim_loss / num_batches
        avg_std_loss = total_std_loss / num_batches
        avg_cov_loss = total_cov_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches

        return avg_loss, avg_sim_loss, avg_std_loss, avg_cov_loss, avg_reward_loss

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
        
        # Add optional model state dicts if they exist
        if self.vicreg_loss is not None:
            checkpoint['vicreg_loss_state_dict'] = self.vicreg_loss.state_dict()
        if self.reward_predictor is not None:
            checkpoint['reward_predictor_state_dict'] = self.reward_predictor.state_dict()
        
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
            
            # Save optional models if they exist
            if self.vicreg_loss is not None:
                torch.save(self.vicreg_loss.state_dict(), self.checkpoint_dir / "best_vicreg_loss.pth")
            if self.reward_predictor is not None:
                torch.save(self.reward_predictor.state_dict(), self.checkpoint_dir / "best_reward_predictor.pth")
            
    def train(self):
        """Run the complete training loop."""
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
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, sim_loss, std_loss, cov_loss, reward_loss = self.train_epoch()

            # Validate
            val_loss, val_sim_loss, val_std_loss, val_cov_loss, val_reward_loss = self.validate_epoch()

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

            if self.use_vicreg:
                log_dict.update({
                    "train_sim_loss": sim_loss,
                    "train_std_loss": std_loss,
                    "train_cov_loss": cov_loss,
                    "val_sim_loss": val_sim_loss,
                    "val_std_loss": val_std_loss,
                    "val_cov_loss": val_cov_loss
                })

            if self.use_reward_predictor:
                log_dict.update({
                    "train_reward_loss": reward_loss,
                    "val_reward_loss": val_reward_loss
                })
                
            if wandb.run is not None:
                wandb.log(log_dict)
                
            # Terminal output (minimal)
            if val_loss is not None:
                # Base message
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                
                # Add VICReg losses if enabled
                if self.use_vicreg:
                    message += f", Train Sim {sim_loss:.6f}, Train Std {std_loss:.6f}, Train Cov {cov_loss:.6f}"
                    message += f", Val Sim {val_sim_loss:.6f}, Val Std {val_std_loss:.6f}, Val Cov {val_cov_loss:.6f}"
                
                # Add reward losses if enabled
                if self.use_reward_predictor:
                    message += f", Train Reward {reward_loss:.6f}, Val Reward {val_reward_loss:.6f}"
                
                print(message)
            else:
                # Base message for no validation
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: N/A"
                
                # Add VICReg losses if enabled
                if self.use_vicreg:
                    message += f", Train Sim {sim_loss:.6f}, Train Std {std_loss:.6f}, Train Cov {cov_loss:.6f}"
                
                # Add reward loss if enabled
                if self.use_reward_predictor:
                    message += f", Train Reward {reward_loss:.6f}"
                
                print(message)
                
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss)
            
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
