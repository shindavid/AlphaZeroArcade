from alphazero.logic.run_params import RunParams
from alphazero.logic import constants
from alphazero.logic.custom_types import ClientData, ClientId, ClientType, ThreadId
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.training_params import TrainingParams
from games.index import get_game_spec
from util.logging_util import get_logger
from util.sqlite3_util import ConnectionPool


from dataclasses import dataclass
import socket
import sys
import threading
from typing import Callable, List


logger = get_logger()


@dataclass
class LoopControllerParams:
    port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    cuda_device: str = 'cuda:0'
    model_cfg: str = 'default'

    @staticmethod
    def create(args) -> 'LoopControllerParams':
        return LoopControllerParams(
            port=args.port,
            cuda_device=args.cuda_device,
            model_cfg=args.model_cfg,
        )

    @staticmethod
    def add_args(parser):
        defaults = LoopControllerParams()
        group = parser.add_argument_group('LoopController options')

        group.add_argument('--port', type=int,
                           default=defaults.port,
                           help='LoopController port (default: %(default)s)')
        group.add_argument('--cuda-device',
                           default=defaults.cuda_device,
                           help='cuda device used for network training (default: %(default)s)')
        group.add_argument('-m', '--model-cfg', default=defaults.model_cfg,
                           help='model config (default: %(default)s)')

    def add_to_cmd(self, cmd: List[str]):
        defaults = LoopControllerParams()
        if self.port != defaults.port:
            cmd.extend(['--port', str(self.port)])
        if self.cuda_device != defaults.cuda_device:
            cmd.extend(['--cuda-device', self.cuda_device])
        if self.model_cfg != defaults.model_cfg:
            cmd.extend(['--model-cfg', self.model_cfg])


class LoopControlData:
    """
    Data owned by LoopController, and accessed via various subcontrollers.
    """

    def __init__(self, params: LoopControllerParams, training_params: TrainingParams,
                 run_params: RunParams):
        self.params = params
        self.training_params = training_params
        self.run_params = run_params

        self.organizer = DirectoryOrganizer(run_params)
        self.game_spec = get_game_spec(run_params.game)

        self.clients_db_conn_pool = ConnectionPool(
            self.organizer.clients_db_filename, constants.CLIENTS_TABLE_CREATE_CMDS)
        self.self_play_db_conn_pool = ConnectionPool(
            self.organizer.self_play_db_filename, constants.SELF_PLAY_TABLE_CREATE_CMDS)
        self.training_db_conn_pool = ConnectionPool(
            self.organizer.training_db_filename, constants.TRAINING_TABLE_CREATE_CMDS)

        self.server_socket = None

        self._shutdown_actions = []
        self._shutdown_in_progress = False
        self._child_thread_error_flag = threading.Event()

        self._client_data_list: List[ClientData] = []
        self._client_data_lock = threading.Lock()

    @property
    def game(self) -> str:
        return self.run_params.game

    def add_client(self, client_data: ClientData):
        with self._client_data_lock:
            self._client_data_list.append(client_data)

    def signal_error(self):
        self._child_thread_error_flag.set()

    def error_signaled(self):
        return self._child_thread_error_flag.is_set()

    @property
    def model_cfg(self):
        return self.params.model_cfg

    def init_server_socket(self):
        # TODO: retry-loop on bind() in case of temporary EADDRINUSE
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setblocking(True)
        self.server_socket.bind(('0.0.0.0', self.params.port))
        self.server_socket.listen()

    def get_clients(self, ctype: ClientType, shared_gpu: bool=False) -> List[ClientData]:
        """
        Returns a list of all client datas of the given type.

        If shared_gpu is True, then only localhost clients that are using the same cuda device
        are returned.
        """
        with self._client_data_lock:
            clients = [c for c in self._client_data_list if c.client_type == ctype]
        if shared_gpu:
            clients = [c for c in clients if c.is_on_localhost() and
                        c.cuda_device == self.params.cuda_device]
        return clients

    def get_single_client_data(self, ctype: ClientType) -> ClientData:
        """
        Returns a single client data of the given type.
        """
        data_list = self.get_clients(ctype)
        assert len(data_list) > 0, f'No clients of type {ctype} connected'
        return data_list[0]

    def active(self):
        return not self._shutdown_in_progress

    def shutdown(self, code):
        self._shutdown_in_progress = True
        logger.info(f'Shutting down (rc={code})...')

        for action in self._shutdown_actions:
            action()
        if self.server_socket:
            self.server_socket.close()
        sys.exit(code)

    def register_shutdown_action(self, action: Callable[[], None]):
        self._shutdown_actions.append(action)

    def remove_client(self, client_id: ClientId):
        with self._client_data_lock:
            self._client_data_list = [
                c for c in self._client_data_list if c.client_id != client_id]

    def close_db_conns(self, thread_id: ThreadId):
        for pool in [self.clients_db_conn_pool, self.training_db_conn_pool, self.self_play_db_conn_pool]:
            pool.close_connections(thread_id)
