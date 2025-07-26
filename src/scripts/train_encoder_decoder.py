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
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, init_decoder, load_config
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device
from src.utils.scheduler_utils import create_lr_scheduler, step_scheduler


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
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = nn.L1Loss()  # Mean Absolute Error for reconstruction
        
        # Data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Logging and checkpointing
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("weights/encoder_decoder")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # MAE pretraining configuration
        self.load_mae_weights = self.training_config.get('load_mae_weights', False)
        self.mae_weights_path = "weights/mae_pretraining/best_checkpoint.pth"
        self.freeze_encoder = self.training_config.get('freeze_encoder', False)
        self.freeze_decoder = self.training_config.get('freeze_decoder', False)
        
        # Plotting configuration
        self.plot_frequency = 5  # Plot every 5 epochs
        self.plot_dir = "evaluation_plots/decoder_plots/encoder_decoder"
        self.validation_sample_indices = None  # Will be set once data is loaded
        
    def initialize_models(self):
        """Initialize encoder, predictor, and decoder models."""
        # Initialize all models
        self.encoder = init_encoder(self.config_path).to(self.device)
        self.predictor = init_predictor(self.config_path).to(self.device)
        self.decoder = init_decoder(self.config_path).to(self.device)
            
        # Load MAE pretrained weights if configured
        if self.load_mae_weights:
            self.load_mae_pretrained_weights()
            
        # Freeze encoder and decoder if configured
        if self.freeze_encoder:
            self.freeze_model(self.encoder, "encoder")
        if self.freeze_decoder:
            self.freeze_model(self.decoder, "decoder")
            
        # Print model parameter counts and frozen status
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        predictor_params = sum(p.numel() for p in self.predictor.parameters()) 
        predictor_trainable = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        print(f"Encoder: {encoder_params} parameters ({encoder_trainable} trainable)")
        print(f"Predictor: {predictor_params} parameters ({predictor_trainable} trainable)")  
        print(f"Decoder: {decoder_params} parameters ({decoder_trainable} trainable)")
            
    def load_mae_pretrained_weights(self):
        """Load pretrained weights from MAE checkpoint."""
        mae_checkpoint_path = Path(self.mae_weights_path)
        
        if not mae_checkpoint_path.exists():
            print(f"Warning: MAE checkpoint not found at {mae_checkpoint_path}")
            print("Proceeding with randomly initialized weights...")
            return
            
        print(f"Loading MAE pretrained weights from {mae_checkpoint_path}")
        
        try:
            checkpoint = torch.load(mae_checkpoint_path, map_location=self.device)
            
            # Load encoder weights
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                print("✓ Loaded pretrained encoder weights")
            else:
                print("Warning: No encoder_state_dict found in MAE checkpoint")
                
            # Load decoder weights  
            if 'decoder_state_dict' in checkpoint:
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                print("✓ Loaded pretrained decoder weights")
            else:
                print("Warning: No decoder_state_dict found in MAE checkpoint")
                
        except Exception as e:
            print(f"Error loading MAE checkpoint: {e}")
            print("Proceeding with randomly initialized weights...")
            
    def freeze_model(self, model, model_name):
        """Freeze all parameters of a model."""
        for param in model.parameters():
            param.requires_grad = False
        print(f"✓ Frozen {model_name} parameters")
        
    def get_trainable_parameters(self):
        """Get all trainable parameters from all models."""
        trainable_params = []
        
        # Add parameters that require gradients from all models
        trainable_params.extend([p for p in self.encoder.parameters() if p.requires_grad])
        trainable_params.extend([p for p in self.predictor.parameters() if p.requires_grad])
        trainable_params.extend([p for p in self.decoder.parameters() if p.requires_grad])
            
        return trainable_params
        
    def initialize_optimizer(self):
        """Initialize the AdamW optimizer and learning rate scheduler for all trainable parameters."""
        # Collect only trainable parameters from all models
        trainable_params = []
        
        # Add encoder parameters if not frozen
        if not self.freeze_encoder:
            trainable_params.extend(list(self.encoder.parameters()))
        else:
            # If frozen, only add parameters that require gradients (should be empty)
            trainable_params.extend([p for p in self.encoder.parameters() if p.requires_grad])
            
        # Add predictor parameters (always trainable)
        trainable_params.extend(list(self.predictor.parameters()))
        
        # Add decoder parameters if not frozen
        if not self.freeze_decoder:
            trainable_params.extend(list(self.decoder.parameters()))
        else:
            # If frozen, only add parameters that require gradients (should be empty)
            trainable_params.extend([p for p in self.decoder.parameters() if p.requires_grad])
        
        # Filter to only parameters that require gradients
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found! Check model freezing configuration.")
        
        print(f"Optimizing {len(trainable_params)} parameter groups")
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
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
        
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
        """
        Perform a single training step with end-to-end reconstruction.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Tuple of (reconstruction_loss, total_loss) for this batch
        """
        state, next_state, action, _ = batch

        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        action = action[:, -1]  # Use only the last action in the sequence
        action = action.to(self.device)
        
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
        
        # 5. Backward pass through entire model graph
        total_loss.backward()
        # Gradient clipping if configured
        if self.gradient_clipping is not None:
            # Include only trainable parameters in gradient clipping
            trainable_params = self.get_trainable_parameters()
            torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clipping)
        self.optimizer.step()
        
        return reconstruction_loss.item(), total_loss.item()
        
    def validate_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
        """
        Perform a single validation step with end-to-end reconstruction.
        
        Args:
            batch: Tuple containing (state, next_state, action, reward)
            
        Returns:
            Tuple of (reconstruction_loss, total_loss) for this batch
        """
        state, next_state, action, _ = batch

        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)

        action = action[:, -1]  # Use only the last action in the sequence
        action = action.to(self.device)
        
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
            
        return reconstruction_loss.item(), total_loss.item()
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_reconstruction_loss, average_total_loss) for the epoch
        """
        self.encoder.train()
        self.predictor.train()
        self.decoder.train()
        
        total_reconstruction_loss = 0.0
        total_total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            reconstruction_loss, batch_total_loss = self.train_step(batch)
            total_reconstruction_loss += reconstruction_loss
            total_total_loss += batch_total_loss
            
            # Log batch loss to wandb (minimal terminal output)
            if wandb.run is not None:
                log_dict = {
                    "batch_reconstruction_loss": reconstruction_loss,
                    "batch_total_loss": batch_total_loss,
                    "batch": batch_idx
                }
                wandb.log(log_dict)
                
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_total_loss = total_total_loss / num_batches
        return avg_reconstruction_loss, avg_total_loss
        
    def validate_epoch(self) -> Optional[Tuple[float, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_reconstruction_loss, average_total_loss) for the epoch, or None if no validation data
        """
        if self.val_dataloader is None:
            return None
            
        self.encoder.eval()
        self.predictor.eval()
        self.decoder.eval()
        
        total_reconstruction_loss = 0.0
        total_total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            reconstruction_loss, batch_total_loss = self.validate_step(batch)
            total_reconstruction_loss += reconstruction_loss
            total_total_loss += batch_total_loss
            
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_total_loss = total_total_loss / num_batches
        return avg_reconstruction_loss, avg_total_loss
        
    def plot_validation_predictions(self, epoch: int):
        """Generate validation plots for the current epoch if needed."""
        # Import here to avoid issues with module-level imports
        from src.utils.plot import plot_validation_samples, get_random_validation_samples, should_plot_validation
        
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
        
        # Set models to eval mode
        self.encoder.eval()
        self.predictor.eval()
        self.decoder.eval()
        
        # Get a validation batch
        val_iter = iter(self.val_dataloader)
        batch = next(val_iter)
        state, next_state, action, _ = batch
        
        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action[:, -1].to(self.device)  # Use only the last action
        
        with torch.no_grad():
            # Forward pass through the models
            z_state = self.encoder(state)
            z_next_pred = self.predictor(z_state, action)
            next_state_reconstructed = self.decoder(z_next_pred)
        
        # Generate plots
        plot_validation_samples(
            true_next_states=next_state,
            predicted_next_states=next_state_reconstructed,
            epoch=epoch,
            sample_indices=self.validation_sample_indices,
            output_dir=self.plot_dir,
            model_name="encoder_decoder"
        )
        
    def save_checkpoint(self, epoch: int, train_reconstruction_loss: float, train_total_loss: float,
                       val_reconstruction_loss: Optional[float], val_total_loss: Optional[float]):
        """
        Save model checkpoints for all models.
        
        Args:
            epoch: Current epoch number
            train_reconstruction_loss: Training reconstruction loss for this epoch
            train_total_loss: Training total loss for this epoch
            val_reconstruction_loss: Validation reconstruction loss for this epoch (if available)
            val_total_loss: Validation total loss for this epoch (if available)
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_reconstruction_loss': train_reconstruction_loss,
            'train_total_loss': train_total_loss,
            'val_reconstruction_loss': val_reconstruction_loss,
            'val_total_loss': val_total_loss,
            'config': self.config
        }
        
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
            
    def train(self):
        """Run the complete end-to-end training loop."""
        print("🚀 Starting Encoder-Decoder training...")
        
        # Print configuration summary
        print("Configuration:")
        print(f"  - Load MAE weights: {self.load_mae_weights}")
        if self.load_mae_weights:
            print(f"  - MAE weights path: {self.mae_weights_path}")
        print(f"  - Freeze encoder: {self.freeze_encoder}")
        print(f"  - Freeze decoder: {self.freeze_decoder}")
        print(f"  - Epochs: {self.num_epochs}, Batch size: {self.batch_size}, LR: {self.learning_rate}")
        print(f"  - Device: {self.device}")
        print()
        
        # Initialize everything
        self.initialize_models()
        self.initialize_optimizer()
        self.load_data()
        
        # Initialize wandb if configured
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            # Create a descriptive run name based on configuration
            run_name_suffix = "pretrained" if self.load_mae_weights else "scratch"
            frozen_parts = []
            if self.freeze_encoder:
                frozen_parts.append("enc")
            if self.freeze_decoder:
                frozen_parts.append("dec")
            frozen_suffix = f"-frozen-{'-'.join(frozen_parts)}" if frozen_parts else ""
            
            wandb.init(
                project=wandb_config.get('project', 'encoder-decoder-training'),
                entity=wandb_config.get('entity'),
                name=f"encoder-decoder-{run_name_suffix}{frozen_suffix}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_reconstruction_loss, train_total_loss = self.train_epoch()
            
            # Validate
            val_results = self.validate_epoch()
            if val_results is not None:
                val_reconstruction_loss, val_total_loss = val_results
            else:
                val_reconstruction_loss, val_total_loss = None, None
            
            # Generate validation plots if needed
            self.plot_validation_predictions(epoch)
            
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
                
            if wandb.run is not None:
                wandb.log(log_dict)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_reconstruction_loss, train_total_loss,
                               val_reconstruction_loss, val_total_loss)
            
            # Step learning rate scheduler
            if self.lr_scheduler is not None:
                # Use validation reconstruction loss for plateau scheduler, otherwise step normally
                step_scheduler(self.lr_scheduler, val_reconstruction_loss)

            # Terminal output
            if val_reconstruction_loss is not None:
                # Base message with reconstruction and total losses
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Recon: {train_reconstruction_loss:.6f}, Val Recon: {val_reconstruction_loss:.6f}, Train Total: {train_total_loss:.6f}, Val Total: {val_total_loss:.6f}"
                
                print(message)
            else:
                # Base message for no validation with reconstruction and total losses
                message = f"Epoch {epoch+1}/{self.num_epochs} - Train Recon: {train_reconstruction_loss:.6f}, Val Recon: N/A, Train Total: {train_total_loss:.6f}, Val Total: N/A"
                
                print(message)
            
        if wandb.run is not None:
            wandb.finish()


def main():
    """Main function for standalone script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Encoder-Decoder models end-to-end')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    parser.add_argument('--load-mae-weights', action='store_true',
                       help='Load pretrained MAE weights (overrides config)')
    parser.add_argument('--mae-weights-path', type=str, 
                       help='Path to MAE checkpoint file (overrides config)')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder parameters (overrides config)')
    parser.add_argument('--freeze-decoder', action='store_true',
                       help='Freeze decoder parameters (overrides config)')
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = EncoderDecoderTrainer(config_path=args.config)
    
    # Override MAE settings from command line if provided
    if args.load_mae_weights:
        trainer.load_mae_weights = True
    if args.mae_weights_path:
        trainer.mae_weights_path = args.mae_weights_path
    if args.freeze_encoder:
        trainer.freeze_encoder = True
    if args.freeze_decoder:
        trainer.freeze_decoder = True
    
    trainer.train()


if __name__ == "__main__":
    main()
