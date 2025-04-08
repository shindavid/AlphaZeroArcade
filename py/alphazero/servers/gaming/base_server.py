from alphazero.logic.build_params import BuildParams
from alphazero.logic.constants import DEFAULT_REMOTE_PLAY_PORT
from alphazero.logic.custom_types import ClientRole
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.logic.signaling import register_standard_server_signals
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.session_data import SessionData
from util.logging_util import LoggingParams
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util import subprocess_util

from abc import abstractmethod
from dataclasses import dataclass
import logging
import subprocess
import threading
from typing import Optional


logger = logging.getLogger(__name__)

@dataclass
class ServerConstants:
  server_name: str
  worker_name: str
  server_role: ClientRole
  worker_role: ClientRole


class BaseServer:
    SERVER_CONSTANTS: ServerConstants

    def __init__(self, params: BaseParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        self._params = params
        self._build_params = build_params
        self._session_data = SessionData(params, logging_params, build_params)
        self._shutdown_manager = ShutdownManager()
        self._running = False
        self._proc: Optional[subprocess.Popen] = None

        self._shutdown_manager.register(lambda: self._shutdown())
        register_standard_server_signals(ignore_sigint=params.ignore_sigint)

    def run(self):
        try:
            threading.Thread(target=self._main_loop, name='main_loop', daemon=True).start()
            self._shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('Caught Ctrl-C')
        finally:
            self._shutdown_manager.shutdown()

    def _shutdown(self):
        logger.info('Shutting down %s...', self.__class__.SERVER_CONSTANTS.server_name)
        try:
            self._session_data.socket.close()
        except:
            pass

        if self._proc is not None:
            try:
                self._proc.terminate()
                subprocess_util.wait_for(self._proc, expected_return_code=None)
                logger.info('Terminated %s process %s', self.__class__.SERVER_CONSTANTS.worker_name ,self._proc.pid)
            except:
                pass
        logger.info('Eval server shutdown complete!')

    def _main_loop(self):
        try:
            self._init_socket()
            self._send_handshake()
            self._recv_handshake()

            threading.Thread(target=self._recv_loop, daemon=True).start()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _init_socket(self):
        self._session_data.init_socket()

    def _send_handshake(self):
        self._session_data.send_handshake(self.__class__.SERVER_CONSTANTS.server_role)

    def _recv_handshake(self):
        self._session_data.recv_handshake(self.__class__.SERVER_CONSTANTS.server_role)

    def _recv_loop(self):
        try:
            self._send_ready()
            while True:
                msg = self._session_data.socket.recv_json()
                if self._handle_msg(msg):
                    break
        except SocketRecvException:
            logger.warning('Encountered SocketRecvException in recv_loop(). '
                        'Loop controller likely shut down.')
            self._shutdown_manager.request_shutdown(0)
        except SocketSendException:
            # Include exc_info in send-case because it's a bit more unexpected
            logger.warning('Encountered SocketSendException in recv_loop(). '
                        'Loop controller likely shut down.', exc_info=True)
            self._shutdown_manager.request_shutdown(0)
        except:
            logger.error('Unexpected error in recv_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _send_ready(self):
        data = { 'type': 'ready', }
        self._session_data.socket.send_json(data)

    def _handle_msg(self, msg: JsonDict) -> bool:
        logger.debug('eval-server received json message: %s', msg)

        msg_type = msg['type']
        if msg_type == 'match-request':
            self._handle_match_request(msg)
        elif msg_type == 'file-transfer':
            self._session_data.receive_files(msg['files'])
        elif msg_type == 'quit':
            self._quit()
            return True
        else:
            raise Exception('Unknown message type: %s', msg_type)
        return False

    def _handle_match_request(self, msg: JsonDict):
        thread = threading.Thread(target=self._run_match, args=(msg,), daemon=True,
                                  name='run-match')
        thread.start()

    def _quit(self):
        logger.info('Received quit command')
        self._shutdown_manager.request_shutdown(0)

    def _run_match(self, msg: JsonDict):
        try:
            self._run_match_helper(msg)
        except:
            logger.error('Unexpected error in run-match:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    @abstractmethod
    def _run_match_helper(self, msg: JsonDict):
        """
        Subclasses should implement this method to:
        1. run match requested in msg
        2. send match result back to loop controller
        """
        raise NotImplementedError('Subclasses must implement _run_match_helper()')