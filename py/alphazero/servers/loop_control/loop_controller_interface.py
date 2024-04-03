from .directory_organizer import DirectoryOrganizer
from .params import LoopControllerParams

from alphazero.logic.custom_types import ClientConnection, ClientRoleOrRoles, DisconnectHandler, \
    Domain, Generation, GpuId, MsgHandler, ShutdownAction
from alphazero.logic.training_params import TrainingParams
from games.game_spec import GameSpec
from util.socket_util import JsonDict
from util.sqlite3_util import DatabaseConnectionPool

import abc
import socket
from typing import List, Optional


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

    @abc.abstractproperty
    def socket(self) -> socket.socket:
        pass

    @abc.abstractproperty
    def training_gpu_id(self) -> GpuId:
        pass

    @abc.abstractproperty
    def game_spec(self) -> GameSpec:
        pass

    @abc.abstractproperty
    def organizer(self) -> DirectoryOrganizer:
        pass

    @abc.abstractproperty
    def params(self) -> LoopControllerParams:
        pass

    @abc.abstractproperty
    def training_params(self) -> TrainingParams:
        pass

    @abc.abstractproperty
    def clients_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @abc.abstractproperty
    def self_play_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @abc.abstractproperty
    def training_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @abc.abstractproperty
    def ratings_db_conn_pool(self) -> DatabaseConnectionPool:
        pass

    @abc.abstractmethod
    def latest_gen(self) -> Generation:
        pass

    @abc.abstractmethod
    def get_connections(self, gpu_id: Optional[GpuId] = None,
                        role: Optional[ClientRoleOrRoles] = None) -> List[ClientConnection]:
        pass

    @abc.abstractmethod
    def acquire_training_gpu_lock(self):
        pass

    @abc.abstractmethod
    def release_training_gpu_lock(self):
        pass

    @abc.abstractmethod
    def acquire_gpu_lock(self, domain: Domain, gpu_id: GpuId):
        pass

    @abc.abstractmethod
    def release_gpu_lock(self, domain: Domain, gpu_id: GpuId):
        pass

    @abc.abstractmethod
    def mark_as_idle(self, domain: Domain, gpu_id: GpuId):
        pass

    @abc.abstractmethod
    def wait_until_gpu_priority_lost(self, domain: Domain, gpu_id: GpuId):
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
                         disconnect_handler: Optional[DisconnectHandler] = None):
        pass

    @abc.abstractmethod
    def handle_new_self_play_positions(self, n_augmented_positions: int):
        pass

    @abc.abstractmethod
    def handle_new_model(self, gen: Generation):
        pass

    @abc.abstractmethod
    def handle_log_msg(self, msg: JsonDict, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def handle_worker_exit(self, msg: JsonDict, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def broadcast_weights(self, conns: List[ClientConnection], gen: Generation):
        pass

    @abc.abstractmethod
    def handle_pause_ack(self, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def handle_unpause_ack(self, conn: ClientConnection):
        pass

    @abc.abstractmethod
    def notify_of_new_rating(self):
        pass
