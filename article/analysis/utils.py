import os
import logging
from pathlib import PosixPath


def setup_logger(
        file_path: PosixPath = None,
        level: int = 20,
        msg_format: str = None
):
    msg_format = msg_format or '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    root_logger = logging.getLogger('analysis')
    root_logger.setLevel(level)
    root_logger.propagate = False
    formatter = logging.Formatter(msg_format, datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # Handle all messages from the logger (not set the handler level)
    root_logger.addHandler(console)
    if file_path:
        os.makedirs(file_path.parent, exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
