"""
Configuration utilities for loading and managing project configuration.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from yaml file.
    
    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in project root.
    
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        # Default to config.yaml in project root (two levels up from utils)
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
