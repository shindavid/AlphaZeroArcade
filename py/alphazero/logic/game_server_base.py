from alphazero.logic.common_params import CommonParams
from alphazero.logic.custom_types import ClientType
from alphazero.logic import constants
from alphazero.logic.directory_organizer import DirectoryOrganizer
from game_index import get_game_spec
from util.logging_util import get_logger
from util.py_util import sha256sum
from util.repo_util import Repo
from util.socket_util import JsonDict, recv_file, recv_json, send_json

import abc
from dataclasses import dataclass, fields
import os
import socket
import sys
import threading
import time


logger = get_logger()


@dataclass
class GameServerBaseParams:
    loop_controller_host: str = 'localhost'
    loop_controller_port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    cuda_device: str = 'cuda:0'

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
        return group


class GameServerBase:
    """
    Common base class for SelfPlayServer and RatingsServer. Contains shared logic for
    interacting with the LoopController and for running games.
    """

    def __init__(self, params: GameServerBaseParams, common_params: CommonParams,
                 client_type: ClientType):
        self.organizer = DirectoryOrganizer(common_params)
        self.game_spec = get_game_spec(common_params.game)
        self.loop_controller_host = params.loop_controller_host
        self.loop_controller_port = params.loop_controller_port
        self.cuda_device = params.cuda_device
        self.client_type = client_type

        self.loop_controller_socket = None
        self.child_process = None
        self.client_id = None
        self.shutdown_code = None

    @property
    def binary_path(self):
        return os.path.join(Repo.root(), '.runtime', self.game_spec.name)

    def init_socket(self):
        loop_controller_address = (self.loop_controller_host, self.loop_controller_port)
        loop_controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        loop_controller_socket.connect(loop_controller_address)

        self.loop_controller_socket = loop_controller_socket

    def run(self):
        self.init_socket()
        try:
            self.send_handshake()
            self.recv_handshake()

            threading.Thread(target=self.recv_loop, daemon=True).start()
            self.error_detection_loop()
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
            logger.error(f'Unexpected error in {func.__name__}():', exc_info=True)
            self.shutdown_code = 1

    def run_func_in_new_thread(self, func, *, args=(), kwargs=None):
        """
        Launches self.run_func(args=args, kwargs=kwargs) in a separate thread.
        """
        kwargs = {'args': args, 'kwargs': kwargs}
        threading.Thread(target=self.run_func, args=(func,), kwargs=kwargs, daemon=True).start()

    def error_detection_loop(self):
        while True:
            time.sleep(1)
            if self.child_process is not None and self.child_process.poll() is not None:
                if self.child_process.returncode != 0:
                    logger.error(f'Child process exited with code {self.child_process.returncode}')
                    self.shutdown_code = 1

            if self.shutdown_code is not None:
                break

    def send_handshake(self):
        data = {
            'type': 'handshake',
            'role': self.client_type.value,
            'start_timestamp': time.time_ns(),
        }

        send_json(self.loop_controller_socket, data)

    def recv_handshake(self):
        data = recv_json(self.loop_controller_socket, timeout=1)
        assert data['type'] == 'handshake_ack', data

        self.client_id = data['client_id']
        logger.info(f'Received client id assignment: {self.client_id}')

        runtime_dir = os.path.join(Repo.root(), '.runtime')
        assets = data['assets']
        requested_assets = []
        for tgt, sha256 in assets:
            loc = os.path.join(runtime_dir, tgt)
            if not os.path.isfile(loc):
                requested_assets.append((tgt, sha256))
                logger.info(f'Requesting asset {tgt} for first time')
                continue

            if sha256sum(loc) != sha256:
                requested_assets.append((tgt, sha256))
                logger.info(f'Re-requesting asset {tgt} due to hash change')
                continue

            logger.debug(f'Asset {tgt} already present')

        for tgt, sha256 in requested_assets:
            data = {
                'type': 'asset_request',
                'asset': tgt,
            }
            send_json(self.loop_controller_socket, data)

            dst = os.path.join(runtime_dir, tgt)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            recv_file(self.loop_controller_socket, dst)
            logger.info(f'Received asset {tgt}')
            if sha256sum(dst) != sha256:
                raise Exception(f'Hash mismatch for asset {tgt}')

        data = {
            'type': 'ready',
        }
        send_json(self.loop_controller_socket, data)

    def recv_loop(self):
        try:
            self.recv_loop_prelude()
            while True:
                msg = recv_json(self.loop_controller_socket)
                if self.handle_msg(msg):
                    break
        except ConnectionError as e:
            if str(e).find('Socket gracefully closed by peer') != -1:
                logger.info(f'Socket gracefully closed by peer')
                self.shutdown_code = 0
                return
            else:
                logger.error(f'Unexpected error in recv_loop():', exc_info=True)
                self.shutdown_code = 1
                return
        except:
            logger.error(f'Unexpected error in recv_loop():', exc_info=True)
            self.shutdown_code = 1
            return

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
        code = self.shutdown_code if self.shutdown_code is not None else 0
        logger.info(f'Shutting down (rc={code})...')
        if self.loop_controller_socket:
            self.loop_controller_socket.close()
        sys.exit(code)

    def quit(self):
        logger.info(f'Received quit command')
        self.shutdown_code = 0
