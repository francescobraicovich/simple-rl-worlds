import yaml
import os
import torchvision.transforms as T
from collections import deque
import torch

def load_config(config_path='config.yaml'):
    """
    Loads a YAML configuration file.
    Args:
        config_path (str): Path to the configuration file.
                           Defaults to 'configs/base_config.yaml'.
    Returns:
        dict: The loaded configuration.
    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handles empty YAML file case
            return {}
        return config
    except yaml.YAMLError as e:
        # Add more context to the YAML parsing error if possible
        error_msg = f"Error parsing YAML file {config_path}: {e}"
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            error_msg += f" at line {mark.line + 1}, column {mark.column + 1}"
        raise yaml.YAMLError(error_msg)


def get_effective_input_channels(config):
    """
    Calculate the effective number of input channels after applying grayscale conversion and frame stacking.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        int: Effective number of input channels
    """
    env_config = config.get('environment', {})
    
    input_channels_per_frame = env_config.get('input_channels_per_frame', 3)
    frame_stack_size = env_config.get('frame_stack_size', 1)
    grayscale_conversion = env_config.get('grayscale_conversion', False)
    
    # Apply grayscale conversion only if input has multiple channels
    if grayscale_conversion and input_channels_per_frame > 1:
        channels_per_frame = 1
    else:
        channels_per_frame = input_channels_per_frame
    
    # Apply frame stacking
    effective_channels = channels_per_frame * frame_stack_size
    
    return effective_channels


def create_image_preprocessing_transforms(config, image_size):
    """
    Create torchvision transforms for image preprocessing based on config.
    
    Args:
        config (dict): Configuration dictionary
        image_size (tuple): Target image size as (height, width)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    env_config = config.get('environment', {})
    input_channels_per_frame = env_config.get('input_channels_per_frame', 3)
    grayscale_conversion = env_config.get('grayscale_conversion', False)
    
    transforms = [T.ToPILImage()]

    
    # Add grayscale conversion only if input has multiple channels and conversion is requested
    if grayscale_conversion and input_channels_per_frame > 1:
        transforms.append(T.Grayscale(num_output_channels=1))
    
    # Add resize and tensor conversion
    transforms.extend([
        T.Resize(image_size),
        T.ToTensor()
    ])
    
    return T.Compose(transforms)


class FrameStackBuffer:
    """
    Buffer for frame stacking. Maintains a rolling buffer of the last N frames.
    """
    
    def __init__(self, frame_stack_size, frame_shape):
        """
        Initialize the frame stack buffer.
        
        Args:
            frame_stack_size (int): Number of frames to stack
            frame_shape (tuple): Shape of individual frames (channels, height, width)
        """
        self.frame_stack_size = frame_stack_size
        self.frame_shape = frame_shape
        self.buffer = deque(maxlen=frame_stack_size)
        
        # Initialize buffer with zeros
        self.reset()
    
    def reset(self):
        """Reset the buffer with zero frames."""
        self.buffer.clear()
        zero_frame = torch.zeros(self.frame_shape)
        for _ in range(self.frame_stack_size):
            self.buffer.append(zero_frame.clone())
    
    def add_frame(self, frame):
        """
        Add a new frame to the buffer.
        
        Args:
            frame (torch.Tensor): Frame to add with shape (channels, height, width)
        """
        if not isinstance(frame, torch.Tensor):
            frame = torch.tensor(frame)
        
        if frame.shape != self.frame_shape:
            raise ValueError(f"Frame shape {frame.shape} doesn't match expected shape {self.frame_shape}")
        
        self.buffer.append(frame.clone())
    
    def get_stacked_frames(self):
        """
        Returns:
            torch.Tensor: Stacked frames with shape (L, C, H, W)
        """
        # Instead of torch.cat on dim=0, use torch.stack to create a new dimension
        return torch.stack(list(self.buffer), dim=0)


def validate_environment_config(config):
    """
    Validate environment configuration parameters for consistency.
    
    Args:
        config (dict): Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    env_config = config.get('environment', {})
    
    input_channels_per_frame = env_config.get('input_channels_per_frame')
    frame_stack_size = env_config.get('frame_stack_size', 1)
    grayscale_conversion = env_config.get('grayscale_conversion', False)
    
    if input_channels_per_frame is None:
        raise ValueError("Configuration error: 'environment.input_channels_per_frame' is not set.")
    
    if input_channels_per_frame not in [1, 3, 4]:
        print(f"Warning: input_channels_per_frame is {input_channels_per_frame}. Common values are 1 (grayscale), 3 (RGB), or 4 (RGBA).")
    
    if frame_stack_size < 1:
        raise ValueError(f"frame_stack_size must be >= 1, got {frame_stack_size}")
    
    # Validate grayscale conversion logic
    if grayscale_conversion and input_channels_per_frame == 1:
        print("Warning: grayscale_conversion is enabled but input_channels_per_frame is already 1. Grayscale conversion will be ignored.")
    
    if grayscale_conversion and input_channels_per_frame not in [3, 4]:
        print(f"Warning: grayscale_conversion is enabled but input_channels_per_frame is {input_channels_per_frame}. Grayscale conversion typically applies to RGB (3) or RGBA (4) inputs.")
    

def get_single_frame_channels(config):
    """
    Calculate the number of channels for a single frame after preprocessing.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        int: Number of channels for a single frame after preprocessing
    """
    env_config = config.get('environment', {})
    
    input_channels_per_frame = env_config.get('input_channels_per_frame', 3)
    grayscale_conversion = env_config.get('grayscale_conversion', False)
    
    # Apply grayscale conversion only if input has multiple channels
    if grayscale_conversion and input_channels_per_frame > 1:
        channels_per_frame = 1
    else:
        channels_per_frame = input_channels_per_frame
    
    return channels_per_frame


# Example of how you might extend it for multiple files later:
# def load_and_merge_configs(primary_config_path, secondary_config_paths=None):
#     config = load_config(primary_config_path)
#     if secondary_config_paths:
#         for path in secondary_config_paths:
#             secondary_config = load_config(path)
#             # Simple dict update, for more complex merges, use a library or custom logic
#             config.update(secondary_config) 
#     return config

