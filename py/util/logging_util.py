from dataclasses import dataclass
import datetime
import logging
import os
import queue
import sys
from typing import List, Optional


DEFAULT_LOGGER_NAME = 'default'


@dataclass
class LoggingParams:
    debug: bool

    @staticmethod
    def create(args) -> 'LoggingParams':
        return LoggingParams(
            debug=bool(args.debug),
        )

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('Logging options')
        group.add_argument('--debug', action='store_true', help='debug mode')

    def add_to_cmd(self, cmd: List[str]):
        if self.debug:
            cmd.append('--debug')


class QueueStream:
    def __init__(self):
        self.log_queue = queue.Queue()

    def write(self, msg):
        self.log_queue.put(msg)

    def flush(self):
        pass  # This could be implemented if needed


class CustomFormatter(logging.Formatter):
    """
    Python's logging module only supports second-level precision. This class allows for finer
    precision.
    """

    # def format(self, record):
    #     record.thread_id = threading.get_ident()
    #     return super(CustomFormatter, self).format(record)

    def formatTime(self, record, datefmt):
        dt = datetime.datetime.fromtimestamp(record.created)
        formatted_time = dt.strftime(datefmt)
        return formatted_time


class NonErrorStreamHandler(logging.StreamHandler):
    def emit(self, record):
        if record.levelno < logging.ERROR:
            super().emit(record)


def configure_logger(*, params: Optional[LoggingParams]=None, filename=None,
                     queue_stream: Optional[QueueStream]=None,
                     prefix='', logger_name=DEFAULT_LOGGER_NAME):
    """
    Configures the logger. A log level of INFO is used by default. If debug is True, then a log
    level of DEBUG is used instead.

    The logger prefixed each line with the current time, and the log level.

    This logger writes error() calls to stderr, and info()/warning() calls to stdout.

    If filename is provided, then the logger will additionally log to the file.

    If queue_stream is provided, then the logger will additionally log to the queue. The expectation
    is that the queue will be consumed by a separate thread.
    """
    debug = params.debug if params else False
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    custom_datefmt = '%Y-%m-%d %H:%M:%S.%f'
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    if prefix:
        fmt = f'{prefix} {fmt}'
    formatter = CustomFormatter(fmt, datefmt=custom_datefmt)
    # formatter = CustomFormatter(
    #     '%(asctime)s [Thread:%(thread_id)s] [%(levelname)s] %(message)s', datefmt=custom_datefmt)

    if filename:
        # Create a file handler and add it to the logger
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if queue_stream:
        queue_handler = logging.StreamHandler(queue_stream)
        queue_handler.setLevel(level)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    non_error_handler = NonErrorStreamHandler(sys.stdout)
    non_error_handler.setLevel(level)
    non_error_handler.setFormatter(formatter)

    logger.addHandler(error_handler)
    logger.addHandler(non_error_handler)

    return logger


def get_logger(logger_name=DEFAULT_LOGGER_NAME):
    return logging.getLogger(logger_name)
