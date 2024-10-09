from alphazero.logic.shutdown_manager import ShutdownManager
from util.logging_util import LoggingParams, QueueStream, configure_logger, get_logger
from util.socket_util import Socket, SocketSendException

import queue
import subprocess
import threading
from typing import Optional


logger = get_logger()


class LogForwarder:
    """
    The LogForwarder is used to forward log messages, both from the current process and from
    subprocesses, to the loop controller for remote logging.

    Usage:

    log_forwarder = LogForwarder(shutdown_manager, logging_params)
    log_forwarder.init_socket(loop_controller_socket)
    log_forwarder.launch()

    proc = subprocess.Popen(...)
    log_forwarder.forward_output('worker', proc)

    Afer the launch() call, the log_forwarder will start forwarding log messages from the
    current process to the loop controller.

    The forward_output() call will wait for proc to complete, raising an Exception if proc
    returns a non-zero error code. The stdout/stderr of proc will be forwarded to the loop
    controller for remote logging. The argument 'worker' is a string that will be sent to the
    loop controller, which it will include in the filename of the log file it creates.
    """

    def __init__(self, shutdown_manager: ShutdownManager, logging_params: LoggingParams):
        self._shutdown_manager = shutdown_manager
        self._logging_params = logging_params
        self._socket: Optional[Socket] = None
        self._logging_queue = QueueStream()
        self._skip_next_returncode_check = False

    def disable_next_returncode_check(self):
        self._skip_next_returncode_check = True

    def set_socket(self, socket: Socket):
        self._socket = socket

    def launch(self):
        configure_logger(params=self._logging_params, queue_stream=self._logging_queue)
        threading.Thread(target=self._log_loop, daemon=True, args=(self._logging_queue.log_queue,),
                         name='log-loop').start()

    def forward_output(self, src: str, proc: subprocess.Popen, stdout_buffer=None,
                       close_remote_log=True):
        """
        Accepts a subprocess.Popen object and forwards its stdout and stderr to the loop controller
        for remote logging. Assumes that the proc was constructed with stdout=subprocess.PIPE and
        stderr=subprocess.PIPE.

        If stdout_buffer is provided, captures the stdout lines in the buffer.

        Note that the relative ordering of stdout and stderr lines is not guaranteed when
        forwarding. This should not be a big deal, since typically proc itself has non-deterministic
        ordering of stdout vs stderr lines.

        Waits for the process to return. Checks the error code and logs the stderr if the process
        returns a non-zero error code.
        """
        proc_log_queue = queue.Queue()
        stderr_buffer = []

        stdout_thread = threading.Thread(target=self._forward_output_thread, daemon=True,
                                         args=(src, proc.stdout, proc_log_queue, stdout_buffer),
                                         name=f'{src}-forward-stdout')
        stderr_thread = threading.Thread(target=self._forward_output_thread, daemon=True,
                                         args=(src, proc.stderr, proc_log_queue, stderr_buffer),
                                         name=f'{src}-forward-stderr')
        forward_thread = threading.Thread(target=self._log_loop, daemon=True,
                                          args=(proc_log_queue, src),
                                          name=f'{src}-log-loop')

        stdout_thread.start()
        stderr_thread.start()
        forward_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        proc_log_queue.put(None)
        forward_thread.join()
        proc.wait()

        try:
            data = {
                'type': 'worker-exit',
                'src': src,
                'close_log': close_remote_log,
            }
            self._socket.send_json(data)
        except SocketSendException:
            pass

        if self._skip_next_returncode_check:
            self._skip_next_returncode_check = False
            return
        if proc.returncode:
            logger.error(f'Process failed with return code {proc.returncode}')
            for line in stderr_buffer:
                logger.error(line.strip())
            raise Exception()

    def _log_loop(self, q: queue.Queue, src: Optional[str] = None):
        try:
            while True:
                line = q.get()
                if line is None:
                    break

                data = {
                    'type': 'log',
                    'line': line,
                }
                if src is not None:
                    data['src'] = src
                self._socket.send_json(data)
        except SocketSendException:
            logger.warning('Loop controller appears to have disconnected, shutting down...')
            self._shutdown_manager.request_shutdown(0)
        except:
            logger.error(f'Unexpected error in log_loop(src={src}):', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _forward_output_thread(self, src: str, stream, q: queue.Queue, buf=None):
        try:
            for line in stream:
                if line is None:
                    break
                q.put(line)
                if buf is not None:
                    buf.append(line)
        except:
            logger.error(f'Unexpected error in _forward_output_thread({src}):', exc_info=True)
            self._shutdown_manager.request_shutdown(1)
