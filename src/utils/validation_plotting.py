"""Shared utility functions for validation plotting across different models."""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import yaml

def get_frame_structure_info(config):
    """
    Get frame structure information from config.
    
    Returns:
        tuple: (frame_stack_size, channels_per_frame, is_grayscale)
    """
    env_config = config.get('environment', {})
    
    input_channels_per_frame = env_config.get('input_channels_per_frame', 3)
    frame_stack_size = env_config.get('frame_stack_size', 1)
    grayscale_conversion = env_config.get('grayscale_conversion', False)
    
    # Determine channels per frame after grayscale conversion
    if grayscale_conversion and input_channels_per_frame > 1:
        channels_per_frame = 1
        is_grayscale = True
    else:
        channels_per_frame = input_channels_per_frame
        is_grayscale = (input_channels_per_frame == 1)
    
    return frame_stack_size, channels_per_frame, is_grayscale

def separate_stacked_frames(stacked_tensor, frame_stack_size, channels_per_frame):
    """
    Separate a stacked frame tensor into individual frames.
    
    Args:
        stacked_tensor: Tensor of shape (C*frame_stack_size, H, W)
        frame_stack_size: Number of stacked frames
        channels_per_frame: Channels per individual frame
        
    Returns:
        list: List of individual frame tensors, each of shape (channels_per_frame, H, W)
    """
    total_channels, H, W = stacked_tensor.shape
    expected_channels = frame_stack_size * channels_per_frame
    
    if total_channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels but got {total_channels}")
    
    frames = []
    for i in range(frame_stack_size):
        start_channel = i * channels_per_frame
        end_channel = start_channel + channels_per_frame
        frame = stacked_tensor[start_channel:end_channel, :, :]
        frames.append(frame)
    
    return frames

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        return None

class ValidationPlotter:
    """Handles consistent validation plotting for encoder-decoder and JEPA decoder models."""
    
    def __init__(self, plot_dir, enable_plotting=True, num_samples=5, random_seed=None, config=None):
        """
        Initialize the validation plotter.
        
        Args:
            plot_dir (str): Directory to save plots
            enable_plotting (bool): Whether plotting is enabled
            num_samples (int): Number of samples to plot per epoch
            random_seed (int, optional): Random seed for reproducible sample selection
            config (dict, optional): Configuration dictionary for frame structure info
        """
        self.plot_dir = plot_dir
        self.enable_plotting = enable_plotting
        self.num_samples = num_samples
        self.random_seed = random_seed
        self._epoch_random_state = None
        
        # Load config if not provided
        if config is None:
            config = load_config()
        self.config = config
        
        # Get frame structure info
        if self.config:
            self.frame_stack_size, self.channels_per_frame, self.is_grayscale = get_frame_structure_info(self.config)
        else:
            # Default values if config not available
            self.frame_stack_size = 1
            self.channels_per_frame = 3
            self.is_grayscale = False
        
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
        
        # Handle the case where we get a stacked frame tensor instead of single frame
        # This can happen if there's a model configuration issue (e.g., old trained model)
        expected_single_frame_channels = getattr(self, 'channels_per_frame', None)
        if expected_single_frame_channels is None:
            # Fallback: assume it's grayscale if self.is_grayscale, otherwise RGB
            expected_single_frame_channels = 1 if self.is_grayscale else 3
            
        if img_np.shape[0] > expected_single_frame_channels:
            print(f"Warning: Got {img_np.shape[0]} channels in image for plotting.")
            print(f"Expected single frame ({expected_single_frame_channels} channels) but got multi-frame tensor.")
            print("This suggests a model output channel mismatch - model may be from old training with different config.")
            print(f"Using first {expected_single_frame_channels} channels only as a temporary fix.")
            print("Consider retraining the model with the current configuration.")
            
            # Extract the expected number of channels for single frame
            img_np = img_np[:expected_single_frame_channels, :, :]
        
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
            current_state: Current state tensor (potentially frame-stacked)
            true_next_state: True next state tensor (single frame)
            predicted_next_state: Predicted next state tensor (single frame)
            epoch: Current epoch number
            sample_num: Sample number (1-indexed)
            model_name: Name to include in plot title
        """
        if not self.enable_plotting:
            return
            
        # Determine layout based on frame stack size
        if self.frame_stack_size == 1:
            # Single frame layout: current, true next, predicted next
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            curr_img = self.process_image_for_plotting(current_state)
            axes[0].imshow(curr_img, cmap='gray' if self.is_grayscale else None)
            axes[0].set_title("Current State (s_t)")
            axes[0].axis('off')
            
        else:
            # Multi-frame layout: stacked frames + true next + predicted next
            total_plots = self.frame_stack_size + 2
            fig, axes = plt.subplots(1, total_plots, figsize=(4 * total_plots, 4))
            
            # Plot each frame in the stack
            try:
                frames = separate_stacked_frames(current_state, self.frame_stack_size, self.channels_per_frame)
                for frame_idx, frame in enumerate(frames):
                    frame_img = self.process_image_for_plotting(frame)
                    axes[frame_idx].imshow(frame_img, cmap='gray' if self.is_grayscale else None)
                    axes[frame_idx].set_title(f"Frame {frame_idx + 1}")
                    axes[frame_idx].axis('off')
            except Exception as e:
                print(f"Warning: Could not separate stacked frames: {e}. Using full tensor.")
                curr_img = self.process_image_for_plotting(current_state)
                axes[0].imshow(curr_img, cmap='gray' if self.is_grayscale else None)
                axes[0].set_title("Current State (s_t)")
                axes[0].axis('off')
                # Hide unused frame axes
                for i in range(1, self.frame_stack_size):
                    axes[i].axis('off')
                    axes[i].set_title("")
        
        # Process and plot true next state
        true_img = self.process_image_for_plotting(true_next_state)
        true_next_axis_idx = self.frame_stack_size if self.frame_stack_size > 1 else 1
        axes[true_next_axis_idx].imshow(true_img, cmap='gray' if self.is_grayscale else None)
        axes[true_next_axis_idx].set_title("True Next State (s_{t+1})")
        axes[true_next_axis_idx].axis('off')
        
        # Process and plot predicted next state
        pred_img = self.process_image_for_plotting(predicted_next_state)
        pred_next_axis_idx = self.frame_stack_size + 1 if self.frame_stack_size > 1 else 2
        axes[pred_next_axis_idx].imshow(pred_img, cmap='gray' if self.is_grayscale else None)
        title = "Predicted Next State"
        if model_name:
            title += f" ({model_name})"
        axes[pred_next_axis_idx].set_title(title)
        axes[pred_next_axis_idx].axis('off')
        
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
    
    # Create plotters with shared random seed and config
    enc_dec_plotter = ValidationPlotter(
        plot_dir=enc_dec_plot_dir,
        enable_plotting=enc_dec_plotting_enabled,
        random_seed=random_seed,
        config=config
    )
    
    jepa_decoder_plotter = ValidationPlotter(
        plot_dir=jepa_plot_dir, 
        enable_plotting=jepa_plotting_enabled,
        random_seed=random_seed,
        config=config
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
