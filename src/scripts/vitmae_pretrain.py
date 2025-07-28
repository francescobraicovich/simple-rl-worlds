#!/usr/bin/env python3
"""
ViT MAE Pretraining Script using Hugging Face Transformers

This script implements self-supervised pretraining using Vision Transformer 
Masked Autoencoder (ViT MAE) from Hugging Face. The script uses the built-in
Trainer and TrainingArguments for simplified training.

The training process:
1. Load video sequences and convert to appropriate format for ViT MAE
2. Use Hugging Face's ViTMAEForPreTraining model
3. Train with built-in masking and reconstruction loss
4. Generate validation plots similar to custom MAE implementation
"""

import os
import sys
import time
import argparse
import torch
import wandb
from pathlib import Path
from transformers import Trainer, TrainingArguments, TrainerCallback

# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_vit_mae, load_config  # noqa: E402
from src.scripts.collect_load_data import DataLoadingPipeline  # noqa: E402
from src.utils.set_device import set_device  # noqa: E402


class ViTMAEDataCollator:
    """
    Data collator for ViT MAE that handles video sequence data.
    Works directly with [B, T, H, W] format where T is treated as channels.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, features):
        """
        Process a batch of video sequences for ViT MAE training.
        
        Args:
            features: List of tuples (state, next_state, action, reward)
            
        Returns:
            Dictionary with pixel_values for ViT MAE
        """
        # Extract states from the batch
        states = []
        #print('length of features:', len(features))  # Debugging line
        #print('shape of features:', [f[0].shape for f in features])  # Debugging line
        for feature in features:
            state, a, r, ns = feature  # Extract state from tuple

            #print('state shape:', state.shape)  # Debugging line
            #print('action shape:', a.shape)  # Debugging line
            #print('reward shape:', r.shape)  # Debugging line
            #print('next_state shape:', ns.shape)  # Debugging line
            # state shape: [T, H, W] where T=4 (sequence length)
            # ViT MAE treats T as channels, which is perfect for our use case
            states.append(state)
        
        # Stack states into a batch: [B, T, H, W]
        pixel_values = torch.stack(states)
        
        return {
            "pixel_values": pixel_values
        }


class ViTMAETrainer:
    """
    Trainer class for ViT MAE pretraining using Hugging Face Transformers.
    
    This class provides a simplified training pipeline using Hugging Face's
    built-in Trainer and TrainingArguments, making the code more concise
    compared to the custom MAE implementation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the ViT MAE trainer.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Device setup using set_device
        self.device = torch.device(set_device())
        
        # Models and data collator
        self.model = None
        self.data_collator = None
        
        # Training arguments (specified in this file as requested)
        self.training_args = TrainingArguments(
            output_dir="./mae-pretrain",
            per_device_train_batch_size=4,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            num_train_epochs=2000,
            learning_rate=1e-4,
            logging_steps=100,
            save_steps=10,
            save_total_limit=2,
            dataloader_drop_last=True,  # Important for consistent batch sizes
            remove_unused_columns=False,  # Keep all columns for our custom data format
            prediction_loss_only=False,
        )
        
        # Data
        self.train_dataset = None
        self.val_dataset = None
        
        # Plotting configuration
        self.plot_frequency = 1  # Plot every epoch
        self.plot_dir = "evaluation_plots/decoder_plots/vitmae_pretrain"
        self.validation_sample_indices = None
        
    def initialize_model(self):
        """Initialize ViT MAE model."""
        # Initialize the ViT MAE model
        self.model = init_vit_mae(self.config_path)
        self.model.to(self.device)
        
        # Initialize data collator (no processor needed)
        self.data_collator = ViTMAEDataCollator()
        
        print(f"Initialized ViT MAE model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def load_data(self):
        """Load training and validation data using DataLoadingPipeline."""
        data_pipeline = DataLoadingPipeline(
            self.training_args.per_device_train_batch_size,
            self.config_path
        )

        # Get training and validation dataloaders
        train_dataloader, val_dataloader = data_pipeline.run_pipeline()
        
        # Convert dataloaders to datasets for Hugging Face Trainer
        self.train_dataset = DataloaderDataset(train_dataloader)
        if val_dataloader is not None:
            self.val_dataset = DataloaderDataset(val_dataloader)
        else:
            self.val_dataset = None
        
        print(f"Loaded {len(self.train_dataset)} training samples")
        if self.val_dataset is not None:
            print(f"Loaded {len(self.val_dataset)} validation samples")
        else:
            print("No validation data available")
            
    def plot_vitmae_validation_predictions(self, epoch: int):
        """Generate ViT MAE validation plots for the current epoch if needed."""
        # Import here to avoid issues with module-level imports
        from src.utils.plot import plot_mae_validation_samples, should_plot_validation
        
        # Check if we should plot for this epoch
        if not should_plot_validation(epoch, self.plot_frequency):
            return
            
        if self.val_dataset is None:
            return
        
        if torch.rand(1).item() < 0.95:
            return
            
        # Get validation sample indices (consistent across epochs)
        if self.validation_sample_indices is None:
            self.validation_sample_indices = list(range(min(5, len(self.val_dataset))))
        
        # Set model to eval mode
        self.model.eval()
        
        # Get validation samples
        val_samples = []
        for idx in self.validation_sample_indices:
            sample = self.val_dataset[idx]
            val_samples.append(sample)
        
        # Process samples through data collator
        batch = self.data_collator(val_samples)
        pixel_values = batch["pixel_values"].to(self.device)
        
        with torch.no_grad():
            # Forward pass through ViT MAE
            outputs = self.model(pixel_values)
            
            # Get reconstructed images from logits
            reconstructed_logits = outputs.logits
            
            # Manual unpatchify: reshape logits to image format
            # logits shape: (batch_size, sequence_length, patch_size ** 2 * num_channels)
            batch_size = reconstructed_logits.shape[0]
            num_patches = reconstructed_logits.shape[1]  # Should be 144 (12*12)
            
            # Reshape to (batch_size, num_patches_h, num_patches_w, patch_size, patch_size, num_channels)
            num_patches_per_side = int(num_patches ** 0.5)  # Should be 12
            patch_size = 7
            num_channels = 4
            
            reconstructed_patches = reconstructed_logits.view(
                batch_size, num_patches_per_side, num_patches_per_side, 
                patch_size, patch_size, num_channels
            )
            
            # Rearrange to (batch_size, num_channels, height, width)
            reconstructed_images = reconstructed_patches.permute(0, 5, 1, 3, 2, 4)
            reconstructed_images = reconstructed_images.contiguous().view(
                batch_size, num_channels, 
                num_patches_per_side * patch_size, 
                num_patches_per_side * patch_size
            )
 
            # Get original images for comparison
            original_images = pixel_values
            
            # Create masked input for visualization using the actual mask
            masked_images = pixel_values.clone()

            # Apply mask to create masked visualization
            # outputs.mask has shape (batch_size, sequence_length) where 1=masked, 0=visible
            mask = outputs.mask  # Shape: (batch_size, num_patches)
            
            # Reshape mask to patch grid: (batch_size, num_patches_h, num_patches_w)
            batch_size = mask.shape[0]
            num_patches_per_side = 12  # 84/7 = 12
            mask_2d = mask.view(batch_size, num_patches_per_side, num_patches_per_side)
            
            # Expand mask to full image resolution
            patch_size = 7
            mask_expanded = mask_2d.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
            # mask_expanded shape: (batch_size, 84, 84)
            
            # Apply mask to all channels
            for b in range(batch_size):
                for c in range(4):  # 4 channels
                    masked_images[b, c][mask_expanded[b] == 1] = 0  # Set masked patches to black
        
        # Convert tensors to appropriate format for plotting
        # Shape conversion: [B, C, H, W] -> [B, T, H, W] (treating C as T)
        original_for_plot = original_images.permute(0, 1, 2, 3)  # Keep as is for now
        reconstructed_for_plot = reconstructed_images.permute(0, 1, 2, 3) if reconstructed_images.dim() == 4 else reconstructed_images
        masked_for_plot = masked_images.permute(0, 1, 2, 3)

        plot_mae_validation_samples(
            masked_frames=masked_for_plot,
            reconstructed_frames=reconstructed_for_plot,
            true_frames=original_for_plot,
            epoch=int(epoch),
            sample_indices=self.validation_sample_indices,
            output_dir=self.plot_dir,
            model_name="vitmae_pretrain"
        )
        
    def train(self):
        """
        Run the complete training loop using Hugging Face Trainer.
        """
        print("🚀 Starting ViT MAE pretraining...")
        print(f"Configuration: {self.training_args.num_train_epochs} epochs, "
              f"batch size {self.training_args.per_device_train_batch_size}, "
              f"lr {self.training_args.learning_rate}")
        print(f"Device: {self.device}")
        
        # Initialize wandb if enabled
        if self.config['wandb']['enabled']:
            wandb.init(
                project=self.config['wandb']['project'],
                name=f"vitmae_pretraining-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    'num_train_epochs': self.training_args.num_train_epochs,
                    'per_device_train_batch_size': self.training_args.per_device_train_batch_size,
                    'learning_rate': self.training_args.learning_rate,
                    'model_type': 'vitmae_pretraining',
                    'device': str(self.device)
                }
            )
            
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
        )
        
        # Add callback for validation plotting
        trainer.add_callback(ValidationPlottingCallback(self))
        
        start_time = time.time()
        
        # Train the model
        trainer.train()
        
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Save the final model
        trainer.save_model("./mae-pretrain/final")
        
        if self.config['wandb']['enabled']:
            wandb.finish()


class DataloaderDataset:
    """
    Wrapper to convert PyTorch DataLoader to Dataset format for Hugging Face Trainer.
    """
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data = []
        
        # Convert dataloader to list of samples
        for batch in dataloader:
            # batch is tuple of (state, next_state, action, reward)
            state, next_state, action, reward = batch
            for i in range(state.shape[0]):  # Iterate through batch dimension
                self.data.append((
                    state[i],      # [T, H, W]
                    next_state[i], # [T, H, W]
                    action[i],     # scalar or [1]
                    reward[i]      # scalar or [1]
                ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class ValidationPlottingCallback(TrainerCallback):
    """
    Callback for generating validation plots during training.
    """
    
    def __init__(self, mae_trainer):
        self.mae_trainer = mae_trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        self.mae_trainer.plot_vitmae_validation_predictions(state.epoch)


def main():
    """Main function for standalone script execution."""
    
    parser = argparse.ArgumentParser(description='Train ViT MAE model for video pretraining')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: project_root/config.yaml)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ViTMAETrainer(config_path=args.config)
    
    # Initialize model and data
    trainer.initialize_model()
    trainer.load_data()
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
