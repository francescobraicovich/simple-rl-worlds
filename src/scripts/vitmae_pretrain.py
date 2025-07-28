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
        for state, next_state, action, reward in features:
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
            per_device_train_batch_size=1,
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
        self.plot_frequency = 20  # Plot every epoch
        self.plot_dir = "evaluation_plots/decoder_plots/vitmae_pretrain"
        self.validation_sample_indices = None
        
        # Gradient monitoring configuration
        self.gradient_check_frequency = 10  # Check gradients every N steps
        self.gradient_threshold = 1e-6  # Threshold for detecting vanishing gradients
        
    def check_initial_gradients(self):
        """
        Perform an initial gradient check before training to detect potential issues early.
        """
        print("\n🔍 Performing initial gradient health check...")
        
        # Set model to training mode
        self.model.train()
        
        # Get a small batch from training data
        if self.train_dataset is None:
            print("⚠️  No training data loaded for gradient check")
            return
            
        # Get first few samples
        sample_batch = [self.train_dataset[i] for i in range(min(2, len(self.train_dataset)))]
        batch = self.data_collator(sample_batch)
        pixel_values = batch["pixel_values"].to(self.device)
        
        # Forward pass
        outputs = self.model(pixel_values)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        gradient_monitor = GradientMonitoringCallback(self)
        grad_stats = gradient_monitor._calculate_gradient_stats(self.model)
        
        print("Initial gradient statistics:")
        print(f"  Mean gradient norm: {grad_stats['mean_grad_norm']:.2e}")
        print(f"  Max gradient norm: {grad_stats['max_grad_norm']:.2e}")
        print(f"  Min gradient norm: {grad_stats['min_grad_norm']:.2e}")
        print(f"  Vanishing gradient ratio: {grad_stats['vanishing_ratio']:.2%}")
        
        # Provide recommendations
        if grad_stats['vanishing_ratio'] > 0.5:
            print("⚠️  WARNING: High vanishing gradient ratio detected in initial check!")
            self._suggest_gradient_fixes(grad_stats)
        elif grad_stats['mean_grad_norm'] > 1.0:
            print("⚠️  WARNING: Large gradients detected - consider gradient clipping")
        else:
            print("✅ Initial gradients look healthy")
            
        # Clear gradients
        self.model.zero_grad()
        
    def _suggest_gradient_fixes(self, grad_stats):
        """Suggest potential fixes for gradient issues."""
        print("\n💡 Suggestions for gradient issues:")
        print("   1. Reduce learning rate (try 1e-5 instead of 1e-4)")
        print("   2. Use gradient clipping (max_grad_norm=1.0)")
        print("   3. Check model initialization")
        print("   4. Consider using a different optimizer (AdamW with weight decay)")
        print("   5. Add gradient accumulation to increase effective batch size")
        if grad_stats['vanishing_ratio'] > 0.8:
            print("   6. CRITICAL: Consider architectural changes or different initialization")
        
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
        #original_for_plot = original_images.permute(0, 1, 2, 3)  # Keep as is for now
        #reconstructed_for_plot = reconstructed_images.permute(0, 1, 2, 3) if reconstructed_images.dim() == 4 else reconstructed_images
        #masked_for_plot = masked_images.permute(0, 1, 2, 3)

        plot_mae_validation_samples(
            masked_frames=masked_images,
            reconstructed_frames=reconstructed_images,
            true_frames=original_images,
            epoch=int(epoch),
            sample_indices=self.validation_sample_indices,
            output_dir=self.plot_dir,
            model_name="vitmae_pretrain"
        )
        
    def train(self):
        """
        Run the complete training loop using Hugging Face Trainer.
        
        This method includes:
        - Gradient health monitoring to detect vanishing/exploding gradients
        - Validation plotting for visual inspection of reconstruction quality
        - Comprehensive logging to wandb (if enabled)
        - Initial gradient check before training starts
        """
        print("🚀 Starting ViT MAE pretraining...")
        print("📊 Gradient monitoring enabled:")
        print(f"   - Check frequency: every {self.gradient_check_frequency} steps")
        print(f"   - Vanishing threshold: {self.gradient_threshold}")
        print("Number of parameters in model:", sum(p.numel() for p in self.model.parameters()))
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
        
        # Add callbacks for validation plotting and gradient monitoring
        trainer.add_callback(ValidationPlottingCallback(self))
        
        # Add gradient monitoring callback
        gradient_monitor = GradientMonitoringCallback(
            self, 
            check_frequency=self.gradient_check_frequency,
            gradient_threshold=self.gradient_threshold
        )
        trainer.add_callback(gradient_monitor)
        
        # Perform initial gradient health check
        self.check_initial_gradients()
        
        start_time = time.time()
        
        # Train the model
        trainer.train()
        
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Print gradient health summary
        print("\n" + gradient_monitor.get_gradient_summary())
        
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


class GradientMonitoringCallback(TrainerCallback):
    """
    Callback for monitoring gradient health during training to detect vanishing gradients.
    """
    
    def __init__(self, mae_trainer, check_frequency=10, gradient_threshold=1e-6):
        """
        Initialize gradient monitoring callback.
        
        Args:
            mae_trainer: The ViTMAETrainer instance
            check_frequency: How often to check gradients (every N steps)
            gradient_threshold: Threshold below which gradients are considered vanishing
        """
        self.mae_trainer = mae_trainer
        self.check_frequency = check_frequency
        self.gradient_threshold = gradient_threshold
        self.gradient_history = []
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        self.step_count += 1
        
        # Only check gradients at specified frequency
        if self.step_count % self.check_frequency != 0:
            return
            
        model = kwargs.get('model')
        if model is None:
            return
            
        # Calculate gradient statistics
        grad_stats = self._calculate_gradient_stats(model)
        
        # Store gradient statistics
        self.gradient_history.append({
            'step': state.global_step,
            'epoch': state.epoch,
            **grad_stats
        })
        
        # Check for vanishing gradients
        self._check_vanishing_gradients(grad_stats, state)
        
        # Log to wandb if enabled
        if self.mae_trainer.config['wandb']['enabled']:
            wandb.log({
                "gradients/mean_grad_norm": grad_stats['mean_grad_norm'],
                "gradients/max_grad_norm": grad_stats['max_grad_norm'],
                "gradients/min_grad_norm": grad_stats['min_grad_norm'],
                "gradients/vanishing_ratio": grad_stats['vanishing_ratio'],
                "gradients/layers_with_vanishing": grad_stats['layers_with_vanishing'],
                "step": state.global_step
            })
    
    def _calculate_gradient_stats(self, model):
        """Calculate comprehensive gradient statistics."""
        grad_norms = []
        layer_grad_norms = {}
        vanishing_count = 0
        total_layers = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Calculate gradient norm for this parameter
                grad_norm = param.grad.data.norm().item()
                grad_norms.append(grad_norm)
                
                # Store per-layer gradient norms
                layer_name = name.split('.')[0]  # Get top-level layer name
                if layer_name not in layer_grad_norms:
                    layer_grad_norms[layer_name] = []
                layer_grad_norms[layer_name].append(grad_norm)
                
                # Check for vanishing gradients
                if grad_norm < self.gradient_threshold:
                    vanishing_count += 1
                    
                total_layers += 1
        
        if not grad_norms:
            return {
                'mean_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'min_grad_norm': 0.0,
                'vanishing_ratio': 1.0,
                'layers_with_vanishing': 0,
                'layer_grad_norms': {}
            }
        
        # Calculate per-layer statistics
        layer_stats = {}
        for layer_name, norms in layer_grad_norms.items():
            layer_stats[layer_name] = {
                'mean': sum(norms) / len(norms),
                'max': max(norms),
                'min': min(norms)
            }
        
        return {
            'mean_grad_norm': sum(grad_norms) / len(grad_norms),
            'max_grad_norm': max(grad_norms),
            'min_grad_norm': min(grad_norms),
            'vanishing_ratio': vanishing_count / total_layers if total_layers > 0 else 1.0,
            'layers_with_vanishing': vanishing_count,
            'layer_grad_norms': layer_stats
        }
    
    def _check_vanishing_gradients(self, grad_stats, state):
        """Check for vanishing gradients and issue warnings."""
        vanishing_ratio = grad_stats['vanishing_ratio']
        mean_grad_norm = grad_stats['mean_grad_norm']
        
        # Issue warnings based on gradient health
        if vanishing_ratio > 0.5:
            print(f"⚠️  WARNING: High vanishing gradient ratio at step {state.global_step}: "
                  f"{vanishing_ratio:.2%} of layers have gradients < {self.gradient_threshold}")
        
        if mean_grad_norm < self.gradient_threshold:
            print(f"⚠️  WARNING: Very small mean gradient norm at step {state.global_step}: "
                  f"{mean_grad_norm:.2e}")
        
        if vanishing_ratio > 0.8:
            print(f"🚨 CRITICAL: Severe vanishing gradients detected at step {state.global_step}! "
                  f"Consider adjusting learning rate, model architecture, or initialization.")
        
        # Log detailed layer information if many layers have vanishing gradients
        if vanishing_ratio > 0.3:
            print(f"Gradient details at step {state.global_step}:")
            print(f"  Mean: {mean_grad_norm:.2e}, Max: {grad_stats['max_grad_norm']:.2e}, "
                  f"Min: {grad_stats['min_grad_norm']:.2e}")
            
            # Show which layers have the smallest gradients
            if grad_stats['layer_grad_norms']:
                sorted_layers = sorted(
                    grad_stats['layer_grad_norms'].items(),
                    key=lambda x: x[1]['mean']
                )
                print("  Layers with smallest gradients:")
                for layer_name, stats in sorted_layers[:3]:  # Show top 3 worst
                    print(f"    {layer_name}: mean={stats['mean']:.2e}, "
                          f"min={stats['min']:.2e}")
    
    def get_gradient_summary(self):
        """Get a summary of gradient health throughout training."""
        if not self.gradient_history:
            return "No gradient data collected"
        
        # Calculate overall statistics
        all_means = [entry['mean_grad_norm'] for entry in self.gradient_history]
        all_vanishing_ratios = [entry['vanishing_ratio'] for entry in self.gradient_history]
        
        summary = f"""
Gradient Health Summary:
========================
Total checks: {len(self.gradient_history)}
Mean gradient norm - Average: {sum(all_means)/len(all_means):.2e}, Range: [{min(all_means):.2e}, {max(all_means):.2e}]
Vanishing ratio - Average: {sum(all_vanishing_ratios)/len(all_vanishing_ratios):.2%}, Max: {max(all_vanishing_ratios):.2%}
"""
        return summary


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
