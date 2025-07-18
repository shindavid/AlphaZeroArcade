import atexit
from dataclasses import dataclass, field
import datetime
import logging
from logging.handlers import QueueHandler, QueueListener
import os
import queue
import signal
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
        group.add_argument('--debug', action='store_true', help='enable debug logging')
        group.add_argument('--debug-module', type=str, nargs='+', default=[],
                           help='specific module(s) to enable debug logging for. Example: '
                                '--debug-module util.sqlite3_util alphazero.servers.gaming.session_data')

    def add_to_cmd(self, cmd: List[str]):
        if self.debug:
            cmd.append('--debug')
        if self.debug_module:
            cmd.append('--debug-module')
            for module in self.debug_module:
                cmd.append(module)


class CustomFormatter(logging.Formatter):
    """
    Python's logging module only supports second-level precision. This class allows for finer
    precision.
    """
    def formatTime(self, record, datefmt):
        dt = datetime.datetime.fromtimestamp(record.created)
        formatted_time = dt.strftime(datefmt)
        return formatted_time


# Module-level queue and listener
_log_queue = queue.Queue(-1)
_listener: QueueListener | None = None


def _shutdown_logging():
    """
    Flush and stop the logging listener, then shutdown the logging system.
    We temporarily ignore SIGINT/SIGTERM so that your signal handlers
    (which raise exceptions) can't fire in the middle of shutdown.
    """
    global _listener

    # 1) Temporarily ignore SIGINT and SIGTERM
    old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
    old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        if _listener:
            try:
                _listener.stop()   # this will inject the sentinel and join the thread
            except Exception:
                pass
    finally:
        # 2) Restore whatever handlers you had before
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    # 3) Shutdown the rest of the logging system
    logging.shutdown()

# Ensure clean shutdown on normal exit or SIGINT/SIGTERM
atexit.register(_shutdown_logging)
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, lambda *args, **kw: _shutdown_logging())


def configure_logger(*, params: Optional[LoggingParams]=None, filename=None,
                     mode='a', prefix=''):
    """
    Configures the logger. A log level of INFO is used by default. If debug is True, then a log
    level of DEBUG is used instead.

    The logger prefixed each line with the current time, and the log level.

    This logger writes error() calls to stderr, and info()/warning() calls to stdout.

    If filename is provided, then the logger will additionally log to the file, using the specified
    mode: 'a' (default) or 'w'.

    Besides these user-visible options, this function serves an important purpose. In python, if you
    mix logging.getLogger() with threads, you can get deadlocks. See:

    https://bugs.python.org/issue6721

    This function sets up a QueueHandler and QueueListener to prevent this deadlock. To get this
    benefit, you must call this function before any threads are created. Note that you can call this
    function a second time to reconfigure the logger, if for example the filename you want to pass
    into this function is not known until after a thread is spawned.
    """
    global _listener

    # Determine log level
    level = logging.DEBUG if (params and params.debug) else logging.INFO

    # Build formatter
    custom_datefmt = '%Y-%m-%d %H:%M:%S.%f'
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    if prefix:
        fmt = f'{prefix} {fmt}'
    formatter = CustomFormatter(fmt, datefmt=custom_datefmt)

    # Build actual handlers list
    handlers: list[logging.Handler] = []
    if filename:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # stderr for ERROR+
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    handlers.append(error_handler)

    # stdout for BELOW ERROR
    non_error_handler = logging.StreamHandler(sys.stdout)
    non_error_handler.setLevel(logging.DEBUG)
    non_error_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    non_error_handler.setFormatter(formatter)
    handlers.append(non_error_handler)

    # Reconfigure root logger to use queue
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    # Attach the QueueHandler
    queue_handler = QueueHandler(_log_queue)
    root.addHandler(queue_handler)

    # Set per-module debug levels if requested
    if params and params.debug_module:
        for module in params.debug_module:
            logging.getLogger(module).setLevel(logging.DEBUG)

    # Start or restart the listener
    if _listener:
        _listener.stop()
    _listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
    _listener.start()
