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
    
def gram_matrix(y):
    """
    Compute the Gram matrix of a batch of feature maps.
    Used for style loss calculation.
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram