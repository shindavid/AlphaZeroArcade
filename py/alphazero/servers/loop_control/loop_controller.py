from .client_connection_manager import ClientConnectionManager
from .database_connection_manager import DatabaseConnectionManager
from .directory_organizer import DirectoryOrganizer
from .params import LoopControllerParams
from .loop_controller_interface import LoopControllerInterface
from .ratings_manager import RatingsManager
from .remote_logging_manager import RemoteLoggingManager
from .self_play_manager import SelfPlayManager
from .shutdown_manager import ShutdownManager
from .training_manager import TrainingManager
from .worker_manager import WorkerManager

from alphazero.logic import constants
from alphazero.logic.custom_types import ClientConnection, ClientRole, GpuInfo, \
    DisconnectHandler, Generation, MsgHandler, ShutdownAction
from alphazero.logic.run_params import RunParams
from alphazero.logic.training_params import TrainingParams
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util.sqlite3_util import DatabaseConnectionPool


import socket
import threading
from typing import List, Optional


logger = get_logger()


class LoopController(LoopControllerInterface):
    """
    The LoopController is a server that acts as the "brains" of the AlphaZero system. It performs
    the neural network training, and also coordinates the activity of external servers.

    Internally, a LoopController maintains a set of managers, each of which is responsible for a
    different aspect of the system. The managers do not directly interact with each other; instead,
    they interact with the LoopController, which acts as a central hub for the system. This
    architecture is intended to make the system more modular and easier to understand. Otherwise,
    the system would be a tangled mess of interdependencies between the various managers.
    """
    def __init__(self, params: LoopControllerParams, training_params: TrainingParams,
                 run_params: RunParams):
        self._game_spec = get_game_spec(run_params.game)
        self._params = params
        self._training_params = training_params
        self._training_gpu_info = GpuInfo(constants.LOCALHOST_IP, params.cuda_device)
        self._socket: Optional[socket.socket] = None
        self._organizer = DirectoryOrganizer(run_params)

        self._shutdown_manager = ShutdownManager()
        self._client_connection_manager = ClientConnectionManager(self)
        self._database_connection_manager = DatabaseConnectionManager(self)
        self._training_manager = TrainingManager(self)
        self._self_play_manager = SelfPlayManager(self)
        self._ratings_manager = RatingsManager(self)
        self._worker_manager = WorkerManager(self)
        self._remote_logging_manager = RemoteLoggingManager(self)

        # TODO: move these into managers

    def run(self):
        """
        Entry-point into the LoopController.
        """
        try:
            threading.Thread(target=self._main_loop, name='main_loop', daemon=True).start()
            self._shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('Caught Ctrl-C')
        finally:
            self._shutdown_manager.shutdown()

    @property
    def socket(self) -> socket.socket:
        if self._socket is None:
            raise Exception('socket not initialized')
        return self._socket

    @property
    def training_gpu_info(self) -> GpuInfo:
        return self._training_gpu_info

    @property
    def game_spec(self) -> GameSpec:
        return self._game_spec

    @property
    def organizer(self) -> DirectoryOrganizer:
        return self._organizer

    @property
    def params(self) -> LoopControllerParams:
        return self._params

    @property
    def training_params(self) -> TrainingParams:
        return self._training_params

    @property
    def clients_db_conn_pool(self) -> DatabaseConnectionPool:
        return self._database_connection_manager.clients_db_conn_pool

    @property
    def self_play_db_conn_pool(self) -> DatabaseConnectionPool:
        return self._database_connection_manager.self_play_db_conn_pool

    @property
    def training_db_conn_pool(self) -> DatabaseConnectionPool:
        return self._database_connection_manager.training_db_conn_pool

    @property
    def ratings_db_conn_pool(self) -> DatabaseConnectionPool:
        return self._database_connection_manager.ratings_db_conn_pool

    def get_connections(self, role: ClientRole,
                        gpu_info: Optional[GpuInfo]=None) -> List[ClientConnection]:
        return self._client_connection_manager.get(role, gpu_info)

    def register_shutdown_action(self, action: ShutdownAction):
        self._shutdown_manager.register(action)

    def request_shutdown(self, return_code: int):
        self._shutdown_manager.request_shutdown(return_code)

    def handle_new_client_connnection(self, conn: ClientConnection):
        """
        Dispatches to a manager to handle a new client connection. The manager will spawn a new
        thread.
        """
        client_role = conn.client_role

        if client_role == ClientRole.SELF_PLAY_SERVER:
            self._self_play_manager.add_server(conn)
        elif client_role == ClientRole.SELF_PLAY_WORKER:
            self._self_play_manager.add_worker(conn)
        elif client_role == ClientRole.RATINGS_SERVER:
            self._ratings_manager.add_server(conn)
        elif client_role == ClientRole.RATINGS_WORKER:
            self._ratings_manager.add_worker(conn)
        else:
            raise Exception(f'Unknown client type: {client_role}')

    def launch_recv_loop(self, msg_handler: MsgHandler, conn: ClientConnection, thread_name: str,
                         disconnect_handler: Optional[DisconnectHandler] = None):
        """
        Launches a daemon thread that loops, receiving json messages from conn and calling
        msg_handler(conn, msg) for each message.

        Catches and logs client-disconnection exceptions. Includes a full stack-trace for the more
        uncommon case where the disconnect is detected during a send operation, and a shorter
        single-line message for the more common case where the disconnect is detected during a recv
        operation.

        Signals an error for other types of exceptions; this will cause the entire process to shut
        down.
        """
        threading.Thread(target=self._launch_recv_loop_inner, name=thread_name,
                         args=(msg_handler, disconnect_handler, conn, thread_name),
                         daemon=True).start()

    def handle_new_self_play_positions(self, n_augmented_positions: int):
        self._training_manager.handle_new_self_play_positions(n_augmented_positions)

    def handle_new_model(self, gen: Generation):
        self._worker_manager.handle_new_model(gen)
        self._self_play_manager.handle_new_model(gen)
        self._ratings_manager.handle_new_model(gen)

    def handle_log_msg(self, msg: JsonDict, conn: ClientConnection):
        self._remote_logging_manager.handle_log_msg(msg, conn)

    def reload_weights(self, conns: List[ClientConnection], gen: Generation):
        self._worker_manager.reload_weights(conns, gen)

    def pause_workers(self, gpu_info: GpuInfo):
        self._worker_manager.pause(gpu_info)

    def handle_pause_ack(self, conn: ClientConnection):
        self._worker_manager.handle_pause_ack(conn)

    def _launch_recv_loop_inner(
            self, msg_handler: MsgHandler, disconnect_handler: DisconnectHandler,
            conn: ClientConnection, thread_name: str):
        try:
            while True:
                msg = conn.socket.recv_json()
                if msg_handler(conn, msg):
                    break
        except SocketRecvException:
            logger.warn(
                f'Encountered SocketRecvException in {thread_name} (conn={conn}):')
        except SocketSendException:
            logger.warn(
                f'Encountered SocketSendException in {thread_name} (conn={conn}):',
                exc_info=True)
        except:
            logger.error(
                f'Unexpected error in {thread_name} (conn={conn}):', exc_info=True)
            self._shutdown_manager.request_shutdown(1)
        finally:
            try:
                if disconnect_handler is not None:
                    disconnect_handler(conn)
                self._handle_disconnect(conn)
            except:
                logger.error(
                    f'Error handling disconnect in {thread_name} (conn={conn}):',
                    exc_info=True)
                self._shutdown_manager.request_shutdown(1)

    def _handle_disconnect(self, conn: ClientConnection):
        logger.info(f'Handling disconnect: {conn}...')
        self._client_connection_manager.remove(conn)
        self._database_connection_manager.close_db_conns(threading.get_ident())
        self._remote_logging_manager.handle_disconnect(conn)
        self._worker_manager.handle_disconnect(conn)
        conn.socket.close()

    def _init_socket(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setblocking(True)
        self._socket.bind(('0.0.0.0', self.params.port))
        self._socket.listen()

        self.register_shutdown_action(lambda: self._socket.close())

    def _main_loop(self):
        try:
            logger.info('Performing LoopController setup...')
            self._organizer.makedirs()
            self._init_socket()
            self._training_manager.setup()
            self._client_connection_manager.start()

            self._self_play_manager.wait_for_gen0_completion()
            self._training_manager.train_gen1_model_if_necessary()

            while True:
                self._training_manager.wait_until_enough_training_data()
                self._training_manager.train_step()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)
