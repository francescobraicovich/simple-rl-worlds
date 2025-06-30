import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return None
    except yaml.YAMLError:
        print("Error: Could not parse config.yaml.")
        return None

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
        stacked_tensor: Tensor of shape (Frames, Channels, H, W) or (C*frame_stack_size, H, W) for backward compatibility
        frame_stack_size: Number of stacked frames
        channels_per_frame: Channels per individual frame
        
    Returns:
        list: List of individual frame tensors, each of shape (channels_per_frame, H, W)
    """
    # Handle new format: (Frames, Channels, H, W)
    if len(stacked_tensor.shape) == 4 and stacked_tensor.shape[0] == frame_stack_size:
        frames = []
        for i in range(frame_stack_size):
            frame = stacked_tensor[i]  # Shape: (Channels, H, W)
            if frame.shape[0] != channels_per_frame:
                raise ValueError(f"Expected {channels_per_frame} channels per frame but got {frame.shape[0]}")
            frames.append(frame)
        return frames
    
    # Handle old format: (C*frame_stack_size, H, W) for backward compatibility
    elif len(stacked_tensor.shape) == 3:
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
    
    else:
        raise ValueError(f"Unexpected tensor shape: {stacked_tensor.shape}. Expected (Frames, Channels, H, W) or (C*frame_stack_size, H, W)")

def load_training_dataset():
    """
    Loads the training dataset from the path specified in config.yaml.

    Returns:
        A tuple containing the training dataset object, the dataset directory path, and config,
        or (None, None, None) if an error occurs.
    """
    config = load_config()
    if config is None:
        return None, None, None
        
    dataset_dir_from_config = "datasets" # Default
    try:
        dataset_dir_from_config = config.get("data", {}).get("dataset", {}).get("dir", "datasets")
        dataset_filename = config.get("data", {}).get("dataset", {}).get("filename", "car_racing_v3_v2.pth")
        dataset_path = os.path.join(dataset_dir_from_config, dataset_filename)

        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found at {dataset_path}")
            return None, None, None

        data = torch.load(dataset_path, weights_only=False)

        train_dataset = data.get("train_dataset")
        if train_dataset is None:
            print("Error: 'train_dataset' not found in the dataset file.")
            return None, None, None

        return train_dataset, dataset_dir_from_config, config

    except Exception:
        full_path_attempted = os.path.join(dataset_dir_from_config, config.get("data", {}).get("dataset", {}).get("filename", "unknown.pth"))
        print(f"Error: Could not load dataset file at {full_path_attempted}.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

def sample_data_points(dataset, num_samples=50):
    """
    Samples a specified number of data points from the dataset.
    """
    try:
        dataset_len = len(dataset)
    except TypeError:
        print("Error: Dataset does not support len(). Cannot sample.")
        return []

    if dataset_len == 0:
        print("Warning: Dataset is empty. No samples to select.")
        return []

    num_samples_to_select = min(num_samples, dataset_len)
    selected_indices = random.sample(range(dataset_len), num_samples_to_select)

    selected_data = []
    for index in selected_indices:
        try:
            item = dataset[index]
            if len(item) >= 4:
                 selected_data.append(item[:4])
            else:
                print(f"Warning: Item at index {index} has fewer than 4 elements: {item}. Skipping.")
        except IndexError:
            print(f"Error: Could not retrieve item at index {index}. Index out of bounds?")
        except Exception as e:
            print(f"Error retrieving item at index {index}: {e}")
    return selected_data

def process_image_for_plotting(img_data):
    """
    Converts a potential tensor or numpy array into a NumPy array suitable for matplotlib.
    Handles PyTorch tensors (C,H,W or H,W) and NumPy arrays.
    """
    to_pil = ToPILImage()

    if torch.is_tensor(img_data):
        img_tensor = img_data.detach().cpu()
        
        # Handle the case where we get more channels than expected
        if img_tensor.ndim == 3 and img_tensor.shape[0] > 3:
            print(f"Warning: Got {img_tensor.shape[0]} channels in image, using first 3 channels")
            img_tensor = img_tensor[:3, :, :]
        
        # Ensure CHW for ToPILImage if it's H,W (e.g. grayscale)
        if img_tensor.ndim == 2: # H, W
            img_tensor = img_tensor.unsqueeze(0) # Add channel dim: 1, H, W
        # ToPILImage handles normalization for [0,1] or [-1,1] range tensors
        pil_img = to_pil(img_tensor)
        return np.array(pil_img)
    elif isinstance(img_data, np.ndarray):
        # Handle the case where we get more channels than expected
        if img_data.ndim == 3 and img_data.shape[0] > 3:
            print(f"Warning: Got {img_data.shape[0]} channels in image, using first 3 channels")
            img_data = img_data[:3, :, :]
        
        # If it's H,W (grayscale), it's fine for imshow.
        # If it's C,H,W, convert to H,W,C for imshow.
        if img_data.ndim == 3 and img_data.shape[0] in [1, 3, 4]: # C, H, W
             # Check if it's more likely C,H,W or H,W,C by looking at channel dim size
            if img_data.shape[2] not in [1, 3, 4]: # Likely C,H,W
                return img_data.transpose(1, 2, 0)
            else: # Likely H,W,C already
                return img_data
        # If H,W or H,W,C, it's generally fine
        return img_data
    else:
        raise TypeError(f"Unsupported image data type: {type(img_data)}")


def generate_and_save_plots(sampled_points, plot_dir_path, config):
    """
    Generates and saves plots for the sampled state-next_state pairs.
    """
    if not sampled_points:
        print("No sampled points to plot.")
        return

    # Get frame structure information
    frame_stack_size, channels_per_frame, is_grayscale = get_frame_structure_info(config)
    
    total_samples_to_plot = len(sampled_points)
    print(f"Starting to generate {total_samples_to_plot} plots...")
    print(f"Frame structure: {frame_stack_size} frames stacked, {channels_per_frame} channels per frame, grayscale: {is_grayscale}")

    for idx, sample in enumerate(sampled_points):
        state, action, reward, next_state = sample

        try:
            # Handle new format where both current and next states may have frame dimension
            # Extract the actual next state frame from next_state
            if len(next_state.shape) == 4:  # New format: (Frames, Channels, H, W)
                actual_next_state = next_state[-1]  # Last frame
            else:  # Old format: single frame (Channels, H, W)
                actual_next_state = next_state
            
            # Multi-frame layout: all current frames + next state
            total_plots = frame_stack_size + 1
            fig, axes = plt.subplots(1, total_plots, figsize=(4 * total_plots, 4))
            
            # Ensure axes is always a list for consistent indexing
            if total_plots == 1:
                axes = [axes]
            
            # Plot each frame in the current state stack
            try:
                frames = separate_stacked_frames(state, frame_stack_size, channels_per_frame)
                for frame_idx, frame in enumerate(frames):
                    frame_img = process_image_for_plotting(frame)
                    axes[frame_idx].imshow(frame_img, cmap='gray' if is_grayscale else None)
                    axes[frame_idx].set_title(f"Current Frame {frame_idx + 1}")
                    axes[frame_idx].axis('off')
            except Exception as e:
                print(f"Warning: Could not separate stacked frames for sample {idx+1}: {e}. Using fallback display.")
                # Fallback: try to display current state directly
                if len(state.shape) == 4:
                    # New format: use first frame as fallback
                    curr_img = process_image_for_plotting(state[0])
                else:
                    # Old format: use as is
                    curr_img = process_image_for_plotting(state)
                axes[0].imshow(curr_img, cmap='gray' if is_grayscale else None)
                axes[0].set_title("Current State (s_t)")
                axes[0].axis('off')
                # Hide unused frame axes
                for i in range(1, frame_stack_size):
                    if i < len(axes) - 1:  # Make sure we don't hide next state axis
                        axes[i].axis('off')
                        axes[i].set_title("")
            
            # Process and plot next state (last frame from next state stack or single frame)
            next_img = process_image_for_plotting(actual_next_state)
            next_state_axis_idx = frame_stack_size
            axes[next_state_axis_idx].imshow(next_img, cmap='gray' if is_grayscale else None)
            axes[next_state_axis_idx].set_title("Next State (s_{t+1})")
            axes[next_state_axis_idx].axis('off')

            # Add action and reward information
            action_str = str(action.item()) if torch.is_tensor(action) and action.numel() == 1 else str(action)
            reward_val = reward.item() if torch.is_tensor(reward) and reward.numel() == 1 else float(reward)

            fig.suptitle(f"Action: {action_str}, Reward: {reward_val:.4f}", fontsize=14)

            save_path = os.path.join(plot_dir_path, f"sample_{idx+1}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

            if (idx + 1) % 10 == 0 or (idx + 1) == total_samples_to_plot:
                print(f"Saved plot {idx+1}/{total_samples_to_plot} to {save_path}")

        except Exception as e:
            print(f"Error processing or plotting sample {idx+1}: {e}. Skipping this sample.")
            if 'fig' in locals(): # Ensure figure is closed if error happened after its creation
                plt.close(fig)
            continue

if __name__ == "__main__":
    print("Loading training dataset...")
    dataset, dataset_dir, config = load_training_dataset()

    if dataset is not None and dataset_dir is not None and config is not None:
        plot_subdir_name = "state_next_state_plots"
        plot_dir_path = os.path.join(dataset_dir, plot_subdir_name)
        os.makedirs(plot_dir_path, exist_ok=True)
        print(f"Plot directory ensured at: {plot_dir_path}")

        try:
            total_samples_in_dataset = len(dataset)
            print(f"Dataset loaded successfully. Number of samples: {total_samples_in_dataset}")

            sampled_points = sample_data_points(dataset, num_samples=50) # Use the actual variable name
            print(f"Selected {len(sampled_points)} samples for analysis/plotting.")

            # Call generate_and_save_plots with config
            if sampled_points:
                 generate_and_save_plots(sampled_points, plot_dir_path, config) # Pass config
                 print(f"\nAll processing finished. Plots saved to {plot_dir_path}")
            else:
                print("No samples were selected, so no plots generated.")

            # Verification printout (optional, can be removed or kept)
            print("\nVerifying first few sampled data points (details):")
            for i, sample in enumerate(sampled_points[:3]):
                state, action, reward, next_state = sample
                state_shape_info = state.shape if hasattr(state, 'shape') else 'N/A'
                next_state_shape_info = next_state.shape if hasattr(next_state, 'shape') else 'N/A'
                action_display = action.item() if torch.is_tensor(action) and action.numel() == 1 else action
                reward_display = reward.item() if torch.is_tensor(reward) and reward.numel() == 1 else reward
                print(f"  Sample {i+1}: State shape: {state_shape_info}, Action: {action_display}, Reward: {reward_display}, Next State shape: {next_state_shape_info}")

        except TypeError as te:
            print(f"TypeError during processing: {te}. Ensure dataset items are structured as expected.")
        except Exception as e:
            print(f"An unexpected error occurred in the main block: {e}")
    else:
        print("Failed to load dataset, dataset_dir, or config.")
