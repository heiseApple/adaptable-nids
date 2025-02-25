import yaml
from pathlib import Path

def load_config(config_path="../config.yaml"):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: A dictionary containing configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    
    with config_path.open('r') as file:
        config = yaml.safe_load(file)
    return config