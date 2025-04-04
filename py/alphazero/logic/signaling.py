import faulthandler
import logging
import signal
from typing import Optional, Type


logger = logging.getLogger(__name__)


def register_standard_server_signals(ignore_sigint: bool):
    """
    Does the following:

    - Registers a handler for SIGTERM that raises a SystemExit on the first SIGTERM and ignores
      subsequent SIGTERMs.

    - If ignore_sigint is True, then registers a handler to ignore SIGINT. Otherwise, registers a
      handler that raises a KeyboardInterrupt on the first SIGINT and ignores subsequent SIGINTs.

    - Registers a handler for SIGUSR1 that prints a per-thread stack trace. This is useful for
      diagnosing deadlocks. To trigger this, run: kill -s SIGUSR1 <pid>

    When we have a parent script (like run_local.py) that launches one or more child servers,
    we typically want to call this with ignore_sigint=True for the child servers. This is because
    a ctrl-C signal on a parent process propagates to all child processes, and we would rather the
    parent more directly control the shutdown of the child servers.

    In general, we ignore subsequent signals of the same type because each server may have some
    important cleanup to do on shutdown, and we don't want to interrupt that cleanup via an
    impatient user spamming ctrl+C. In the event a server is hanging and the user needs to kill it
    more forcefully, they can always use SIGKILL (kill -9).
    """
    register_signal_exception(signal.SIGTERM,
                              echo_action=lambda: logger.info('Ignoring repeat SIGTERM'))
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    else:
        register_signal_exception(signal.SIGINT, KeyboardInterrupt,
                                  echo_action=lambda: logger.info('Ignoring repeat Ctrl-C'))

    # This line allows us to generate a per-thread stack trace by externally running:
    #
    # kill -s SIGUSR1 <pid>
    #
    # This is useful for diagnosing deadlocks.
    faulthandler.register(signal.SIGUSR1, all_threads=True)

_signal_set = set()


def register_signal_exception(code: signal.Signals, exception_cls: Type[BaseException]=SystemExit,
                              echo_action: Optional[callable]=None):
    """
    Registers a signal handler that raises an exception of the specified class.

    If echo_action is specified, then on "echos" (i.e., a follow-up signal of the same type),
    echo_action is called instead of raising the exception.
    """
    def handler(signum, frame):
        global _signal_set
        if signum in _signal_set and echo_action is not None:
            echo_action()
        else:
            _signal_set.add(signum)
            raise exception_cls(f'Received signal {signum}')

    signal.signal(code, handler)
