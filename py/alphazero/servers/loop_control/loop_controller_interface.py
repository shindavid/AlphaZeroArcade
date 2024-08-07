from .directory_organizer import DirectoryOrganizer
from .gpu_contention_table import GpuContentionTable
from .params import LoopControllerParams

from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientConnection, DisconnectHandler, Generation, GpuId, \
    MsgHandler, ShutdownAction
from shared.training_params import TrainingParams
from games.game_spec import GameSpec
from util.socket_util import JsonDict
from util.sqlite3_util import DatabaseConnectionPool

import abc
import socket
from typing import Callable, List, Optional


class LoopControllerInterface(abc.ABC):
    """
    An abstract base class for LoopController.

    Various members of LoopController hold a reference back to the LoopController object. This
    convenience allows for more fluid communication between the different components of
    LoopController. However, this can lead to circular-import issues if attempting type hinting.
    By type hinting with this abstract base class, we can avoid such issues.

    The various abstract methods and properties are declared here purely to help with IDE
    autocompletion and type hinting.
    """

    @property
    @abc.abstractmethod
    def socket(self) -> socket.socket:
        pass

    @property
    @abc.abstractmethod
    def default_training_gpu_id(self) -> GpuId:
        pass

    @property
    @abc.abstractmethod
    def game_spec(self) -> GameSpec:
        pass

    @property
    @abc.abstractmethod
    def organizer(self) -> DirectoryOrganizer:
        pass

    @property
    @abc.abstractmethod
    def params(self) -> LoopControllerParams:
        pass

    @property
    @abc.abstractmethod
    def training_params(self) -> TrainingParams:
        pass

    @property
    @abc.abstractmethod
    def build_params(self) -> BuildParams:
        pass

    @property
    @abc.abstractmethod
    def clients_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @property
    @abc.abstractmethod
    def self_play_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @property
    @abc.abstractmethod
    def training_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @property
    @abc.abstractmethod
    def ratings_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @abc.abstractmethod
    def latest_gen(self) -> Generation:
        pass

    @abc.abstractmethod
    def get_gpu_lock_table_for_training(self) -> GpuContentionTable:
        pass

    @abc.abstractmethod
    def get_gpu_lock_table(self, gpu_id: GpuId) -> GpuContentionTable:
        pass

    @abc.abstractmethod
    def reset_self_play_locks(self):
        pass

    @abc.abstractmethod
    def register_shutdown_action(self, action: ShutdownAction):
        pass

    @abc.abstractmethod
    def request_shutdown(self, return_code: int):
        pass

    @abc.abstractmethod
    def handle_new_client_connnection(self, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def launch_recv_loop(self, msg_handler: MsgHandler, conn: ClientConnection, thread_name: str,
                         disconnect_handler: Optional[DisconnectHandler] = None,
                         preamble: Optional[Callable[[], None]] = None):
        pass

    @abc.abstractmethod
    def handle_new_self_play_positions(self, n_augmented_positions: int):
        pass

    @abc.abstractmethod
    def handle_new_model(self):
        pass

    @abc.abstractmethod
    def handle_log_msg(self, msg: JsonDict, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def handle_worker_exit(self, msg: JsonDict, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def broadcast_weights(self, conn: ClientConnection, gen: Generation):
        pass

    @abc.abstractmethod
    def set_ratings_priority(self, elevate: bool):
        pass
