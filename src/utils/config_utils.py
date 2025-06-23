import yaml
import os

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

# Example of how you might extend it for multiple files later:
# def load_and_merge_configs(primary_config_path, secondary_config_paths=None):
#     config = load_config(primary_config_path)
#     if secondary_config_paths:
#         for path in secondary_config_paths:
#             secondary_config = load_config(path)
#             # Simple dict update, for more complex merges, use a library or custom logic
#             config.update(secondary_config) 
#     return config

