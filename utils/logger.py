import logging
import os
from pathlib import Path

def setup_logger(log_file_path: str, log_level: int = logging.INFO) -> logging.Logger:
    # Configures a universal logger with console and file handlers.
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger