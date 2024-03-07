from alphazero.logic.common_params import CommonParams
from alphazero.logic import constants
from alphazero.logic.custom_types import ClientData, ClientId, ClientType, ThreadId
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.training_params import TrainingParams
from game_index import get_game_spec
from util.logging_util import get_logger
from util.py_util import sha256sum
from util.repo_util import Repo
from util.sqlite3_util import ConnectionPool


from dataclasses import dataclass
import os
import socket
import sys
import threading
from typing import List


logger = get_logger()


@dataclass
class LoopControllerParams:
    port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    cuda_device: str = 'cuda:0'
    model_cfg: str = 'default'
    binary_path: str = None

    @staticmethod
    def create(args) -> 'LoopControllerParams':
        return LoopControllerParams(
            port=args.port,
            cuda_device=args.cuda_device,
            model_cfg=args.model_cfg,
            binary_path=args.binary_path,
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
        group.add_argument('-b', '--binary-path',
                           help='binary path. Default: last-used binary for this tag. If this is '
                           'the first run for this tag, then target/Release/bin/{game}')

    def add_to_cmd(self, cmd: List[str]):
        defaults = LoopControllerParams()
        if self.port != defaults.port:
            cmd.extend(['--port', str(self.port)])
        if self.cuda_device != defaults.cuda_device:
            cmd.extend(['--cuda-device', self.cuda_device])
        if self.model_cfg != defaults.model_cfg:
            cmd.extend(['--model-cfg', self.model_cfg])
        if self.binary_path:
            cmd.extend(['--binary-path', self.binary_path])


@dataclass
class RuntimeAsset:
    src_path: str
    tgt_path: str
    sha256: str

    @staticmethod
    def make(src_path: str, tgt_path: str):
        return RuntimeAsset(src_path, tgt_path, str(sha256sum(src_path)))


class LoopControlData:
    """
    Data owned by LoopController, and accessed via various subcontrollers.
    """

    def __init__(self, params: LoopControllerParams, training_params: TrainingParams,
                 common_params: CommonParams):
        self.params = params
        self.training_params = training_params
        self.common_params = common_params

        self.organizer = DirectoryOrganizer(common_params)
        self.game_spec = get_game_spec(common_params.game)

        self.binary_path = self._get_binary_path()
        self.binary_asset = RuntimeAsset.make(self.binary_path, self.game_spec.name)
        self.extra_assets = self._make_extra_assets()

        self.clients_db_conn_pool = ConnectionPool(
            self.organizer.clients_db_filename, constants.CLIENTS_TABLE_CREATE_CMDS)
        self.self_play_db_conn_pool = ConnectionPool(
            self.organizer.self_play_db_filename, constants.SELF_PLAY_TABLE_CREATE_CMDS)
        self.training_db_conn_pool = ConnectionPool(
            self.organizer.training_db_filename, constants.TRAINING_TABLE_CREATE_CMDS)

        self.server_socket = None

        self._child_thread_error_flag = threading.Event()

        self._client_data_list: List[ClientData] = []
        self._client_data_lock = threading.Lock()

    def add_client(self, client_data: ClientData):
        with self._client_data_lock:
            self._client_data_list.append(client_data)

    def signal_error(self):
        self._child_thread_error_flag.set()

    def error_signaled(self):
        return self._child_thread_error_flag.is_set()

    @property
    def bins_dir(self):
        return self.organizer.bins_dir

    @property
    def model_cfg(self):
        return self.params.model_cfg

    def _get_binary_path(self):
        if self.params.binary_path:
            return self.params.binary_path

        latest = self.organizer.get_latest_binary()
        if latest is not None:
            return latest

        bin_name = self.game_spec.name
        return os.path.join(Repo.root(), f'target/Release/bin/{bin_name}')

    def _make_extra_assets(self):
        assets = []
        for extra in self.game_spec.extra_runtime_deps:
            src = os.path.join(Repo.root(), extra)
            tgt = os.path.join('extra', os.path.basename(extra))
            assets.append(RuntimeAsset.make(src, tgt))
        return assets

    def init_server_socket(self):
        # TODO: retry-loop on bind() in case of temporary EADDRINUSE
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setblocking(True)
        self.server_socket.bind(('0.0.0.0', self.params.port))
        self.server_socket.listen()

    def get_client_data_list(self, ctype: ClientType) -> List[ClientData]:
        """
        Returns a list of all client datas of the given type.
        """
        with self._client_data_lock:
            return [c for c in self._client_data_list if c.client_type == ctype]

    def get_single_client_data(self, ctype: ClientType) -> ClientData:
        """
        Returns a single client data of the given type.
        """
        data_list = self.get_client_data_list(ctype)
        assert len(data_list) > 0, f'No clients of type {ctype} connected'
        return data_list[0]

    def shutdown(self, code):
        logger.info(f'Shutting down (rc={code})...')

        if self.server_socket:
            self.server_socket.close()
        sys.exit(code)

    def remove_client(self, client_id: ClientId):
        with self._client_data_lock:
            self._client_data_list = [
                c for c in self._client_data_list if c.client_id != client_id]

    def close_db_conns(self, thread_id: ThreadId):
        for pool in [self.clients_db_conn_pool, self.training_db_conn_pool, self.self_play_db_conn_pool]:
            pool.close_connections(thread_id)
