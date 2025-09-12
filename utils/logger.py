import logging
import os

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = None,
    fmt: str = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
) -> logging.Logger:
    
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(level)

    return logger
