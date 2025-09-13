import logging
import os
from config.parser import load_config

def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
) -> logging.Logger:
    
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        # Console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # File handler - use data_path from config
        try:
            config = load_config()
            data_path = config.get('data_path', './data')
            log_file = os.path.join(data_path, 'autopi.log')
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback to console only if config loading fails
            print(f"Warning: Could not set up file logging: {e}")

        logger.setLevel(level)

    return logger
