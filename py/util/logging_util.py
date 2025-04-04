from dataclasses import dataclass, field
import datetime
import logging
import os
import queue
import sys
from typing import List, Optional


@dataclass
class LoggingParams:
    debug: bool
    debug_module: List[str] = field(default_factory=list)

    @staticmethod
    def create(args) -> 'LoggingParams':
        return LoggingParams(
            debug=bool(args.debug),
            debug_module=args.debug_module,
        )

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('Logging options')
        group.add_argument('--debug', action='store_true', help='enable debug loggign')
        group.add_argument('--debug-module', type=str, nargs='+', default=[],
                           help='specific module(s) to enable debug logging for. Example: '
                                '--debug-module=util.sqlite3_util --debug-module=alphazero.servers.gaming.session_data')

    def add_to_cmd(self, cmd: List[str]):
        if self.debug:
            cmd.append('--debug')
        if self.debug_module:
            for module in self.debug_module:
                cmd.append('--debug-module')
                cmd.append(module)


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
                     mode='a', prefix=''):
    """
    Configures the logger. A log level of INFO is used by default. If debug is True, then a log
    level of DEBUG is used instead.

    The logger prefixed each line with the current time, and the log level.

    This logger writes error() calls to stderr, and info()/warning() calls to stdout.

    If filename is provided, then the logger will additionally log to the file, using the specified
    mode: 'a' (default) or 'w'.
    """
    debug = params.debug if params else False
    level = logging.DEBUG if debug else logging.INFO

    custom_datefmt = '%Y-%m-%d %H:%M:%S.%f'
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    if prefix:
        fmt = f'{prefix} {fmt}'
    formatter = CustomFormatter(fmt, datefmt=custom_datefmt)

    handlers = []
    if filename:
        # Create a file handler and add it to the logger
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    non_error_handler = NonErrorStreamHandler(sys.stdout)
    non_error_handler.setLevel(logging.DEBUG)
    non_error_handler.setFormatter(formatter)

    handlers.append(error_handler)
    handlers.append(non_error_handler)

    logging.basicConfig(level=level, handlers=handlers)

    if params.debug_module is not None:
        for module in params.debug_module:
            logging.getLogger(module).setLevel(logging.DEBUG)
