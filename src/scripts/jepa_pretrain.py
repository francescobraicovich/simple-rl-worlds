#!/usr/bin/env python3
"""
Joint Embedding Predictive Architecture (JEPA) Pretraining Script

This script implements self-supervised pretraining using JEPA approach.
The main objective is to train encoder and target encoder models to learn
powerful representations through latent space reconstruction.

The training process:
1. Extract tubelets from video sequences
2. Randomly mask a subset of tubelets
3. Encode masked input through encoder
4. Encode unmasked input through target encoder (EMA updated)
5. Use predictor to predict target encoder output from encoder output
6. Compute L1 loss in the latent space
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

from src.utils.init_models import init_encoder, load_config
from src.models.jepa_pretrain_predictor import JEPAPretrainPredictor
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device
from src.utils.scheduler_utils import create_lr_scheduler, step_scheduler, get_current_lr
from src.utils.mask import extract_tubelets, reassemble_tubelets, generate_random_mask


class JEPATrainer:
    """
    Trainer class for Joint Embedding Predictive Architecture (JEPA) pretraining.
    
    This class handles the complete pretraining pipeline including:
    - Model initialization (encoder, target encoder, predictor)
    - Data loading and preprocessing
    - Tubelet extraction and masking
    - Training and validation loops with latent space reconstruction loss
    - EMA updates for target encoder
    - Optimization of encoder and predictor parameters only
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
        
        # Training parameters (shared with MAE pretraining)
        self.training_config = self.config['training']['pretraining']
        self.num_epochs = self.training_config['num_epochs']
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = self.training_config['learning_rate']
        self.weight_decay = self.training_config['weight_decay']
        self.gradient_clipping = self.training_config.get('gradient_clipping', None)
        self.mask_ratio = self.training_config['mask_ratio']
        
        # EMA decay for target encoder (use from main_loops config or default)
        self.ema_decay = self.config['training']['main_loops'].get('ema_decay', 0.999)
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models
        self.encoder = None
        self.target_encoder = None
        self.predictor = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = nn.L1Loss()  # L1 Loss for latent space reconstruction
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/jepa_pretraining")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_models(self):
        """Initialize encoder, target encoder, and predictor models."""
        # Initialize main encoder
        self.encoder = init_encoder(self.config_path).to(self.device)
        
        # Initialize target encoder as copy of main encoder
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        
        # Initialize predictor
        self.predictor = JEPAPretrainPredictor.from_config(self.config).to(self.device)
        
        # Freeze target encoder (only updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        print(f"Initialized encoder with {sum(p.numel() for p in self.encoder.parameters())} parameters")
        print(f"Initialized target encoder with {sum(p.numel() for p in self.target_encoder.parameters())} parameters")
        print(f"Initialized predictor with {sum(p.numel() for p in self.predictor.parameters())} parameters")
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer and learning rate scheduler for trainable parameters."""
        # Only optimize encoder and predictor parameters (target encoder updated via EMA)
        trainable_params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        
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
            
    def update_target_encoder(self):
        """Update target encoder using exponential moving average (EMA)."""
        with torch.no_grad():
            for target_param, encoder_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * encoder_param.data
            
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
            
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single training step with JEPA latent space reconstruction loss.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Latent space reconstruction loss for this batch
        """
        state, next_state, action, reward = batch

        # Move to device - we use current state for JEPA pretraining
        state = state.to(self.device)  # [B, T, H, W]
        
        # Forward pass through JEPA pipeline
        self.optimizer.zero_grad()
        
        # 1. Extract tubelets from the batch
        tubelets = extract_tubelets(state)  # [B, N, PATCH_T, PATCH_H, PATCH_W]
        B, N, PATCH_T, PATCH_H, PATCH_W = tubelets.shape
        
        # 2. Sample a mask from uniform distribution over tubelet dimension
        mask = generate_random_mask(B, N, self.mask_ratio, self.device)  # [B, N] boolean
        
        # 3. Clone tubelets and zero out masked ones to create masked input
        masked_tubelets = tubelets.clone()
        # Expand mask to match tubelet dimensions
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
        mask_expanded = mask_expanded.expand(-1, -1, PATCH_T, PATCH_H, PATCH_W)  # [B, N, PATCH_T, PATCH_H, PATCH_W]
        masked_tubelets[mask_expanded] = 0.0
        
        # 4. Use reassemble_tubelets to fold masked tubelets back into tensor
        masked_input = reassemble_tubelets(masked_tubelets)  # [B, T, H, W]
        
        # 5. Pass masked input through encoder to get latent representations
        encoder_output = self.encoder(masked_input)  # [B, latent_dim]
        
        # 6. Pass encoder output through predictor to predict target representations
        predicted_target = self.predictor(encoder_output)  # [B, latent_dim]
        
        # 7. Pass unmasked input through target encoder (no gradients)
        with torch.no_grad():
            target_output = self.target_encoder(state)  # [B, latent_dim]
        
        # 8. Compute L1 loss between predicted and target representations in latent space
        loss = self.criterion(predicted_target, target_output)
        
        # 9. Backward pass and optimizer step
        loss.backward()
        
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            trainable_params = list(self.encoder.parameters()) + list(self.predictor.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clipping)
            
        self.optimizer.step()
        
        # 10. Update target encoder via EMA
        self.update_target_encoder()
        
        return loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        Perform a single validation step with JEPA latent space reconstruction loss.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Latent space reconstruction loss for this batch
        """
        state, next_state, action, reward = batch

        # Move to device - we use current state for JEPA pretraining
        state = state.to(self.device)  # [B, T, H, W]
        
        with torch.no_grad():
            # 1. Extract tubelets from the batch
            tubelets = extract_tubelets(state)  # [B, N, PATCH_T, PATCH_H, PATCH_W]
            B, N, PATCH_T, PATCH_H, PATCH_W = tubelets.shape
            
            # 2. Sample a mask from uniform distribution over tubelet dimension
            mask = generate_random_mask(B, N, self.mask_ratio, self.device)  # [B, N] boolean
            
            # 3. Clone tubelets and zero out masked ones to create masked input
            masked_tubelets = tubelets.clone()
            # Expand mask to match tubelet dimensions
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
            mask_expanded = mask_expanded.expand(-1, -1, PATCH_T, PATCH_H, PATCH_W)  # [B, N, PATCH_T, PATCH_H, PATCH_W]
            masked_tubelets[mask_expanded] = 0.0
            
            # 4. Use reassemble_tubelets to fold masked tubelets back into tensor
            masked_input = reassemble_tubelets(masked_tubelets)  # [B, T, H, W]
            
            # 5. Pass masked input through encoder to get latent representations
            encoder_output = self.encoder(masked_input)  # [B, latent_dim]
            
            # 6. Pass encoder output through predictor to predict target representations
            predicted_target = self.predictor(encoder_output)  # [B, latent_dim]
            
            # 7. Pass unmasked input through target encoder
            target_output = self.target_encoder(state)  # [B, latent_dim]
            
            # 8. Compute L1 loss between predicted and target representations in latent space
            loss = self.criterion(predicted_target, target_output)
            
        return loss.item()
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.encoder.train()
        self.predictor.train()
        self.target_encoder.eval()  # Target encoder always in eval mode
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Log batch-level metrics to wandb if enabled
            if self.config['wandb']['enabled']:
                wandb.log({
                    "batch_loss": loss,
                    "batch": batch_idx
                })
                
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
        num_batches = 0
        
        for batch in self.val_dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        return avg_loss
        
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
            'encoder_state_dict': self.encoder.state_dict(),
            'target_encoder_state_dict': self.target_encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'ema_decay': self.ema_decay,
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
            
            # Also save individual model state dicts for easy loading
            torch.save(self.encoder.state_dict(), self.checkpoint_dir / "best_encoder.pth")
            torch.save(self.target_encoder.state_dict(), self.checkpoint_dir / "best_target_encoder.pth")
            torch.save(self.predictor.state_dict(), self.checkpoint_dir / "best_predictor.pth")
                        
    def train(self):
        """
        Run the complete training loop.
        """
        print("🚀 Starting JEPA pretraining...")
        print(f"Configuration: {self.num_epochs} epochs, batch size {self.batch_size}, lr {self.learning_rate}")
        print(f"Mask ratio: {self.mask_ratio}, EMA decay: {self.ema_decay}, Device: {self.device}")
        
        # Initialize wandb if enabled
        if self.config['wandb']['enabled']:
            wandb.init(
                project=self.config['wandb']['project'],
                name=f"jepa_pretraining-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    **self.training_config,
                    'model_type': 'jepa_pretraining',
                    'ema_decay': self.ema_decay,
                    'device': str(self.device)
                }
            )
            
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate_epoch()
            
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
                  f"Train Loss: {train_loss:.6f}{val_loss_str}")            
            # Log epoch-level metrics to wandb if enabled
            if self.config['wandb']['enabled']:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                    "ema_decay": self.ema_decay
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Train JEPA model for video pretraining')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: project_root/config.yaml)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = JEPATrainer(config_path=args.config)
    
    # Initialize models and data
    trainer.initialize_models()
    trainer.initialize_optimizer()
    trainer.load_data()
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
