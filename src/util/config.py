import yaml
import os

def load_config(config_path="../config.yaml"):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: A dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config