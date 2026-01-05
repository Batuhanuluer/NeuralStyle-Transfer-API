import logging
import sys

def setup_logger(name="nst_project"):
    """
    Configures a standard logger for the project.
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add new ones (prevents duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger