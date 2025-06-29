"""Shared utility functions for validation plotting across different models."""

import matplotlib.pyplot as plt
import numpy as np
import os
import random

class ValidationPlotter:
    """Handles consistent validation plotting for encoder-decoder and JEPA decoder models."""
    
    def __init__(self, plot_dir, enable_plotting=True, num_samples=5, random_seed=None):
        """
        Initialize the validation plotter.
        
        Args:
            plot_dir (str): Directory to save plots
            enable_plotting (bool): Whether plotting is enabled
            num_samples (int): Number of samples to plot per epoch
            random_seed (int, optional): Random seed for reproducible sample selection
        """
        self.plot_dir = plot_dir
        self.enable_plotting = enable_plotting
        self.num_samples = num_samples
        self.random_seed = random_seed
        self._epoch_random_state = None
        
    def set_epoch_random_state(self, epoch):
        """Set the random state for consistent sampling across models for a given epoch."""
        if self.random_seed is not None:
            # Create a deterministic seed based on base seed and epoch
            epoch_seed = self.random_seed + epoch
            self._epoch_random_state = np.random.RandomState(epoch_seed)
            random.seed(epoch_seed)
        else:
            self._epoch_random_state = np.random.RandomState()
    
    def select_random_batch_and_samples(self, dataloader, device):
        """
        Select a random batch and random samples from validation dataloader.
        
        Args:
            dataloader: Validation dataloader
            device: Torch device
            
        Returns:
            tuple: (batch_data, selected_indices) where batch_data is (s_t, a_t, r_t, s_t_plus_1)
                   and selected_indices are the indices of samples to plot
        """
        if not self.enable_plotting or not dataloader:
            return None, None
            
        # Select random batch
        total_batches = len(dataloader)
        if self._epoch_random_state is not None:
            batch_idx = self._epoch_random_state.randint(0, total_batches)
        else:
            batch_idx = random.randint(0, total_batches - 1)
        
        # Get the selected batch
        for i, batch in enumerate(dataloader):
            if i == batch_idx:
                s_t, a_t, r_t, s_t_plus_1 = batch
                s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
                
                # Select random samples from the batch
                batch_size = s_t.shape[0]
                num_plot_samples = min(self.num_samples, batch_size)
                
                if self._epoch_random_state is not None:
                    selected_indices = self._epoch_random_state.choice(
                        batch_size, num_plot_samples, replace=False
                    )
                else:
                    selected_indices = np.random.choice(
                        batch_size, num_plot_samples, replace=False
                    )
                
                return (s_t, a_t, r_t, s_t_plus_1), selected_indices
        
        return None, None
    
    def process_image_for_plotting(self, img_tensor):
        """
        Process a tensor image for matplotlib plotting.
        
        Args:
            img_tensor: Torch tensor of shape (C, H, W) or (H, W)
            
        Returns:
            numpy array ready for matplotlib imshow
        """
        img_np = img_tensor.cpu().numpy()
        
        # Handle channel dimension
        if img_np.shape[0] == 1 or img_np.shape[0] == 3:  # C, H, W
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # Handle grayscale
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(axis=2)
        
        # Clip float values
        if img_np.dtype in [np.float32, np.float64]:
            img_np = np.clip(img_np, 0, 1)
            
        return img_np
    
    def save_comparison_plot(self, current_state, true_next_state, predicted_next_state, 
                           epoch, sample_num, model_name=""):
        """
        Save a comparison plot of current state, true next state, and predicted next state.
        
        Args:
            current_state: Current state tensor
            true_next_state: True next state tensor  
            predicted_next_state: Predicted next state tensor
            epoch: Current epoch number
            sample_num: Sample number (1-indexed)
            model_name: Name to include in plot title
        """
        if not self.enable_plotting:
            return
            
        # Process images
        curr_img = self.process_image_for_plotting(current_state)
        true_img = self.process_image_for_plotting(true_next_state) 
        pred_img = self.process_image_for_plotting(predicted_next_state)
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(curr_img)
        axes[0].set_title("Current State (s_t)")
        axes[0].axis('off')
        
        axes[1].imshow(true_img)
        axes[1].set_title("True Next State (s_{t+1})")
        axes[1].axis('off')
        
        axes[2].imshow(pred_img)
        title = "Predicted Next State"
        if model_name:
            title += f" ({model_name})"
        axes[2].set_title(title)
        axes[2].axis('off')
        
        # Save plot
        os.makedirs(self.plot_dir, exist_ok=True)
        plot_filename = os.path.join(self.plot_dir, f"epoch_{epoch}_sample_{sample_num}_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig)
        
    def plot_validation_samples(self, batch_data, selected_indices, predictions, epoch, model_name=""):
        """
        Plot validation samples for a batch.
        
        Args:
            batch_data: Tuple of (s_t, a_t, r_t, s_t_plus_1)
            selected_indices: Indices of samples to plot
            predictions: Predicted next states for the selected samples only
            epoch: Current epoch number
            model_name: Name of the model for titles
        """
        if not self.enable_plotting or batch_data is None or selected_indices is None:
            return
            
        s_t, _, _, s_t_plus_1 = batch_data
        
        for i, batch_idx in enumerate(selected_indices):
            sample_num = i + 1  # 1-indexed sample numbers
            self.save_comparison_plot(
                current_state=s_t[batch_idx],
                true_next_state=s_t_plus_1[batch_idx],
                predicted_next_state=predictions[i],  # Use i (0-4) not batch_idx
                epoch=epoch,
                sample_num=sample_num,
                model_name=model_name
            )
        
        print(f"  Saved {len(selected_indices)} validation image samples to {self.plot_dir}")


class RewardPlotter:
    """Handles validation plotting for reward predictors (MLP and LARP)."""
    
    def __init__(self, plot_dir, model_type="reward", enable_plotting=True):
        """
        Initialize the reward plotter.
        
        Args:
            plot_dir (str): Directory to save plots
            model_type (str): Type of model ("reward" or "larp")
            enable_plotting (bool): Whether plotting is enabled
        """
        self.plot_dir = plot_dir
        self.model_type = model_type
        self.enable_plotting = enable_plotting
        
    def plot_reward_scatter(self, true_rewards, predicted_rewards, epoch, model_name=""):
        """
        Create a scatter plot of true vs predicted rewards.
        
        Args:
            true_rewards: Array of true reward values
            predicted_rewards: Array of predicted reward values
            epoch: Current epoch number
            model_name: Name of the model for titles
        """
        if not self.enable_plotting:
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Convert to numpy arrays if they're tensors
        if hasattr(true_rewards, 'cpu'):
            true_rewards = true_rewards.cpu().numpy()
        if hasattr(predicted_rewards, 'cpu'):
            predicted_rewards = predicted_rewards.cpu().numpy()
        
        # Flatten arrays if needed
        true_rewards = np.array(true_rewards).flatten()
        predicted_rewards = np.array(predicted_rewards).flatten()
        
        # Calculate metrics
        r2 = r2_score(true_rewards, predicted_rewards)
        mse = mean_squared_error(true_rewards, predicted_rewards)
        
        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.scatter(true_rewards, predicted_rewards, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_val = min(true_rewards.min(), predicted_rewards.min())
        max_val = max(true_rewards.max(), predicted_rewards.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('True Rewards')
        ax.set_ylabel('Predicted Rewards')
        title = f'{self.model_type.upper()} - True vs Predicted Rewards (Epoch {epoch})'
        if model_name:
            title += f' - {model_name}'
        ax.set_title(title)
        
        # Add metrics text
        textstr = f'R² = {r2:.4f}\nMSE = {mse:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(self.plot_dir, exist_ok=True)
        plot_filename = os.path.join(self.plot_dir, f"epoch_{epoch}_{self.model_type}_scatter.png")
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig)

def create_shared_validation_plotters(config, main_model_dir, random_seed=42):
    """
    Create validation plotters for encoder-decoder and JEPA decoder with shared random seed.
    
    Args:
        config: Configuration dictionary
        main_model_dir: Main model directory
        random_seed: Shared random seed for consistent sampling
        
    Returns:
        tuple: (enc_dec_plotter, jepa_decoder_plotter)
    """
    # Get plotting configurations
    training_config = config.get('training', {})
    models_config = config.get('models', {})
    
    # Use evaluation_plots directory instead of trained_models
    evaluation_plots_dir = "evaluation_plots"
    
    # Encoder-decoder plotting config
    enc_dec_plotting_enabled = training_config.get('enable_validation_plot', True)
    enc_dec_plot_dir = os.path.join(evaluation_plots_dir, "decoder_plots", "encoder_decoder")
    
    # JEPA decoder plotting config
    jepa_config = models_config.get('jepa', {})
    jepa_decoder_config = jepa_config.get('decoder_training', {})
    jepa_plotting_enabled = jepa_decoder_config.get('enable_validation_plot', True)
    jepa_plot_dir = os.path.join(evaluation_plots_dir, "decoder_plots", "jepa_decoder")
    
    # Create plotters with shared random seed
    enc_dec_plotter = ValidationPlotter(
        plot_dir=enc_dec_plot_dir,
        enable_plotting=enc_dec_plotting_enabled,
        random_seed=random_seed
    )
    
    jepa_decoder_plotter = ValidationPlotter(
        plot_dir=jepa_plot_dir, 
        enable_plotting=jepa_plotting_enabled,
        random_seed=random_seed
    )
    
    return enc_dec_plotter, jepa_decoder_plotter


def create_reward_plotters(config):
    """
    Create reward plotters for both encoder-decoder and JEPA models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary with plotters for different model types
    """
    evaluation_plots_dir = "evaluation_plots"
    
    # Check if reward MLP is enabled
    reward_config = config.get('models', {}).get('reward_predictors', {}).get('reward_mlp', {})
    reward_enabled = reward_config.get('enabled', False)
    
    # Check if LARP is enabled
    larp_config = config.get('models', {}).get('reward_predictors', {}).get('larp', {})
    larp_enabled = larp_config.get('enabled', False)
    
    plotters = {}
    
    if reward_enabled:
        # Create reward MLP plotters for both encoder types
        plotters['reward_enc_dec'] = RewardPlotter(
            plot_dir=os.path.join(evaluation_plots_dir, "reward_plots", "encoder_decoder"),
            model_type="reward",
            enable_plotting=True
        )
        plotters['reward_jepa'] = RewardPlotter(
            plot_dir=os.path.join(evaluation_plots_dir, "reward_plots", "jepa_decoder"),
            model_type="reward", 
            enable_plotting=True
        )
    
    if larp_enabled:
        # Create LARP plotters for both encoder types
        plotters['larp_enc_dec'] = RewardPlotter(
            plot_dir=os.path.join(evaluation_plots_dir, "larp_plots", "encoder_decoder"),
            model_type="larp",
            enable_plotting=True
        )
        plotters['larp_jepa'] = RewardPlotter(
            plot_dir=os.path.join(evaluation_plots_dir, "larp_plots", "jepa_decoder"),
            model_type="larp",
            enable_plotting=True
        )
    
    return plotters
