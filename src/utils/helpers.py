import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path):
    """
    Load project configuration from a YAML file.
    :param config_path: Path to the yaml configuration file.
    :return: Dictionary containing configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise e