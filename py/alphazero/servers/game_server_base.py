from alphazero.logic.custom_types import ClientType
from alphazero.logic import constants
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import LoggingParams, QueueStream, configure_logger, get_logger
from util.repo_util import Repo
from util.socket_util import JsonDict, Socket, SocketRecvException, SocketSendException

import abc
from dataclasses import dataclass, fields
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from typing import Optional


logger = get_logger()


@dataclass
class GameServerBaseParams:
    loop_controller_host: str = 'localhost'
    loop_controller_port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    cuda_device: str = 'cuda:0'
    binary_path: str = None

    @classmethod
    def create(cls, args):
        kwargs = {f.name: getattr(args, f.name) for f in fields(cls)}
        return cls(**kwargs)

    @staticmethod
    def add_args_helper(parser, server_name: str):
        defaults = GameServerBaseParams()
        group = parser.add_argument_group(f'{server_name} options')

        group.add_argument('--loop-controller-host', type=str,
                           default=defaults.loop_controller_host,
                           help='loop-controller host (default: %(default)s)')
        group.add_argument('--loop-controller-port', type=int,
                           default=defaults.loop_controller_port,
                           help='loop-controller port (default: %(default)s)')
        group.add_argument('--cuda-device', default=defaults.cuda_device,
                           help='cuda device (default: %(default)s)')
        group.add_argument('-b', '--binary-path',
                           help='binary path. Default: target/Release/bin/{game}')
        return group


class GameServerBase:
    """
    Common base class for SelfPlayServer and RatingsServer. Contains shared logic for
    interacting with the LoopController and for running games.
    """

    def __init__(self, params: GameServerBaseParams, logging_params: LoggingParams,
                 client_type: ClientType):
        self._game = None
        self._game_spec = None
        self.logging_params = logging_params
        self.params = params
        self.client_type = client_type

        self.logging_queue = QueueStream()
        self.loop_controller_socket: Optional[Socket] = None
        self.client_id = None

        self._shutdown_lock = threading.Lock()
        self._shutdown_code = -1

    @property
    def game(self) -> str:
        if self._game is None:
            raise ValueError('game not set')
        return self._game

    @property
    def game_spec(self) -> GameSpec:
        if self._game_spec is None:
            self._game_spec = get_game_spec(self.game)
        return self._game_spec

    @property
    def loop_controller_host(self):
        return self.params.loop_controller_host

    @property
    def loop_controller_port(self):
        return self.params.loop_controller_port

    @property
    def cuda_device(self):
        return self.params.cuda_device

    @property
    def binary_path(self):
        if self.params.binary_path:
            return self.params.binary_path
        return os.path.join(Repo.root(), 'target/Release/bin', self.game_spec.name)

    def init_socket(self):
        loop_controller_address = (self.loop_controller_host, self.loop_controller_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(loop_controller_address)

        self.loop_controller_socket = Socket(sock)

    def _set_shutdown_code(self, code, func, *args, **kwargs):
        """
        If code is greater than the current shutdown code, sets the shutdown code to code and calls
        func(*args, **kwargs).
        """
        with self._shutdown_lock:
            if self._shutdown_code >= code:
                return
            self._shutdown_code = code
        func(*args, **kwargs)

    def run(self):
        self.init_socket()
        try:
            self.send_handshake()
            self.recv_handshake()

            threading.Thread(target=self.recv_loop, daemon=True).start()
            self.error_detection_loop()
        except KeyboardInterrupt:
            logger.info(f'Caught Ctrl-C')
        except:
            logger.error(f'Unexpected error in run():', exc_info=True)
        finally:
            self.shutdown()

    def run_func(self, func, *, args=(), kwargs=None):
        """
        Runs func(*args, **kwargs) in a try/except block. In case of an exception, logs the
        exception and sets self.shutdown_code to 1.
        """
        try:
            func(*args, **(kwargs or {}))
        except:
            self._set_shutdown_code(1, logger.error, f'Unexpected error in {func.__name__}():',
                                    exc_info=True)

    def run_func_in_new_thread(self, func, *, args=(), kwargs=None):
        """
        Launches self.run_func(args=args, kwargs=kwargs) in a separate thread.
        """
        kwargs = {'args': args, 'kwargs': kwargs}
        threading.Thread(target=self.run_func, args=(func,), kwargs=kwargs, daemon=True).start()

    def error_detection_loop(self):
        while True:
            time.sleep(1)
            if self._shutdown_code >= 0:
                break

    def send_handshake(self):
        data = {
            'type': 'handshake',
            'role': self.client_type.value,
            'start_timestamp': time.time_ns(),
        }

        self.loop_controller_socket.send_json(data)

    def log_loop(self, q: queue.Queue, src: Optional[str]=None):
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
                self.loop_controller_socket.send_json(data)
        except SocketSendException:
            self._set_shutdown_code(
                0, logger.warn, f'Loop controller appears to have disconnected, shutting down...')
        except:
            self._set_shutdown_code(1, logger.error, f'Unexpected error in log_loop():',
                                    exc_info=True)

    def forward_output(self, src: str, proc: subprocess.Popen, stdout_buffer=None):
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
        forward_thread = threading.Thread(target=self.log_loop, daemon=True,
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

        if proc.returncode:
            logger.error(f'Process failed with return code {proc.returncode}')
            for line in stderr_buffer:
                logger.error(line.strip())
            raise Exception()

    def _forward_output_thread(self, src: str, stream, q: queue.Queue, buf=None):
        try:
            for line in stream:
                if line is None:
                    break
                q.put(line)
                if buf is not None:
                    buf.append(line)
        except:
            self._set_shutdown_code(1, logger.error,
                                    f'Unexpected error in _forward_output_thread({src}):',
                                    exc_info=True)

    def recv_handshake(self):
        data = self.loop_controller_socket.recv_json(timeout=1)
        assert data['type'] == 'handshake-ack', data

        self.client_id = data['client_id']
        self._game = data['game']

        configure_logger(params=self.logging_params, queue_stream=self.logging_queue)
        threading.Thread(target=self.log_loop, daemon=True, args=(self.logging_queue.log_queue,),
                         name='log-loop').start()

        logger.info(f'**** Starting {self.client_type.value} ****')
        logger.info(f'Received client id assignment: {self.client_id}')

    def recv_loop(self):
        try:
            self.recv_loop_prelude()
            while True:
                msg = self.loop_controller_socket.recv_json()
                if self.handle_msg(msg):
                    break
        except SocketRecvException:
            self._set_shutdown_code(0, logger.warn,
                                    'Encountered SocketRecvException in recv_loop(). '
                                    'Loop controller likely shut down.')
        except SocketSendException:
            # Include exc_info in send-case because it's a bit more unexpected
            self._set_shutdown_code(0, logger.warn,
                                    'Encountered SocketSendException in recv_loop(). '
                                    'Loop controller likely shut down.', exc_info=True)
        except:
            self._set_shutdown_code(1, logger.error,
                                    f'Unexpected error in recv_loop():', exc_info=True)

    @abc.abstractmethod
    def handle_msg(self, msg: JsonDict) -> bool:
        """
        Handle the message, return True if should break the loop.

        Must override in subclass.
        """
        pass

    def recv_loop_prelude(self):
        """
        Override to do any work after the handshake is complete but before the recv-loop
        starts.
        """
        pass

    def shutdown(self):
        code = max(0, self._shutdown_code)
        logger.info(f'Shutting down (rc={code})...')
        if self.loop_controller_socket:
            self.loop_controller_socket.close()
        sys.exit(code)

    def quit(self):
        logger.info(f'Received quit command')
        self._shutdown_code = 0
