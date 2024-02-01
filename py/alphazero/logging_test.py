import datetime
import logging
from typing import Optional


class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt):
        return datetime.datetime.fromtimestamp(record.created).strftime(datefmt)


def configure_logger(filename: Optional[str]=None):
    # Create a logger
    logger = logging.getLogger(__name__)

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    custom_datefmt = '%Y-%m-%d %H:%M:%S.%f'
    formatter = CustomFormatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt=custom_datefmt)

    if filename:
        # Create a file handler and add it to the logger
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Create a console handler and add it to the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Initialize the logger
logger = configure_logger()

logger.info('This is an info message')
