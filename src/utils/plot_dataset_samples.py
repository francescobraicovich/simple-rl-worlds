import os
import pickle
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def load_training_dataset():
    """
    Loads the training dataset from the path specified in config.yaml.

    Returns:
        A tuple containing the training dataset object and the dataset directory path,
        or (None, None) if an error occurs.
    """
    dataset_dir_from_config = "datasets" # Default
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        dataset_dir_from_config = config.get("dataset_dir", "datasets")
        dataset_filename = config.get("dataset_filename", "car_racing_v3_v2.pkl")
        dataset_path = os.path.join(dataset_dir_from_config, dataset_filename)

        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found at {dataset_path}")
            return None, None

        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        train_dataset = data.get("train_dataset")
        if train_dataset is None:
            print("Error: 'train_dataset' not found in the dataset file.")
            return None, None

        return train_dataset, dataset_dir_from_config

    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return None, None
    except yaml.YAMLError:
        print("Error: Could not parse config.yaml.")
        return None, None
    except pickle.UnpicklingError:
        full_path_attempted = os.path.join(dataset_dir_from_config, config.get("dataset_filename", "unknown.pkl") if 'config' in locals() else "unknown.pkl")
        print(f"Error: Could not unpickle dataset file at {full_path_attempted}.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

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
        # Ensure CHW for ToPILImage if it's H,W (e.g. grayscale)
        if img_tensor.ndim == 2: # H, W
            img_tensor = img_tensor.unsqueeze(0) # Add channel dim: 1, H, W
        # ToPILImage handles normalization for [0,1] or [-1,1] range tensors
        pil_img = to_pil(img_tensor)
        return np.array(pil_img)
    elif isinstance(img_data, np.ndarray):
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


def generate_and_save_plots(sampled_points, plot_dir_path):
    """
    Generates and saves plots for the sampled state-next_state pairs.
    """
    if not sampled_points:
        print("No sampled points to plot.")
        return

    total_samples_to_plot = len(sampled_points)
    print(f"Starting to generate {total_samples_to_plot} plots...")

    for idx, sample in enumerate(sampled_points):
        state, action, reward, next_state = sample

        try:
            np_state_img = process_image_for_plotting(state)
            np_next_state_img = process_image_for_plotting(next_state)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(np_state_img)
            axes[0].set_title("Current State")
            axes[0].axis('off')

            axes[1].imshow(np_next_state_img)
            axes[1].set_title("Next State")
            axes[1].axis('off')

            action_str = str(action.item()) if torch.is_tensor(action) and action.numel() == 1 else str(action)
            reward_val = reward.item() if torch.is_tensor(reward) and reward.numel() == 1 else float(reward)

            fig.suptitle(f"Action: {action_str}, Reward: {reward_val:.4f}", fontsize=14)

            save_path = os.path.join(plot_dir_path, f"sample_{idx+1}.png")
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
    dataset, dataset_dir = load_training_dataset()

    if dataset is not None and dataset_dir is not None:
        plot_subdir_name = "state_next_state_plots"
        plot_dir_path = os.path.join(dataset_dir, plot_subdir_name)
        os.makedirs(plot_dir_path, exist_ok=True)
        print(f"Plot directory ensured at: {plot_dir_path}")

        try:
            total_samples_in_dataset = len(dataset)
            print(f"Dataset loaded successfully. Number of samples: {total_samples_in_dataset}")

            sampled_points = sample_data_points(dataset, num_samples=50) # Use the actual variable name
            print(f"Selected {len(sampled_points)} samples for analysis/plotting.")

            # Call generate_and_save_plots
            if sampled_points:
                 generate_and_save_plots(sampled_points, plot_dir_path) # Use the actual variable name
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
        print("Failed to load dataset or dataset_dir not found.")
