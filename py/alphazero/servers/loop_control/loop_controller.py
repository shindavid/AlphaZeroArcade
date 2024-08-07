from .client_connection_manager import ClientConnectionManager
from .database_connection_manager import DatabaseConnectionManager
from .directory_organizer import DirectoryOrganizer
from .gpu_contention_table import GpuContentionTable
from .params import LoopControllerParams
from .loop_controller_interface import LoopControllerInterface
from .ratings_manager import RatingsManager
from .remote_logging_manager import RemoteLoggingManager
from .self_play_manager import SelfPlayManager
from .training_manager import TrainingManager
from .gpu_contention_manager import GpuContentionManager

from alphazero.logic import constants
from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientConnection, ClientRole, DisconnectHandler, \
    Generation, GpuId, MsgHandler, RatingTag, ShutdownAction
from alphazero.logic.run_params import RunParams
from alphazero.logic.shutdown_manager import ShutdownManager
from shared.training_params import TrainingParams
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketRecvException, SocketSendException, send_file, \
    send_json
from util.sqlite3_util import DatabaseConnectionPool


import faulthandler
import signal
import socket
import threading
from typing import Callable, Dict, List, Optional


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
                 run_params: RunParams, build_params: BuildParams):
        self._game_spec = get_game_spec(run_params.game)
        self._params = params
        self._training_params = training_params
        self._build_params = build_params
        self._default_training_gpu_id = GpuId(constants.LOCALHOST_IP, params.cuda_device)
        self._socket: Optional[socket.socket] = None
        self._organizer = DirectoryOrganizer(run_params)

        self._shutdown_manager = ShutdownManager()
        self._client_connection_manager = ClientConnectionManager(self)
        self._database_connection_manager = DatabaseConnectionManager(self)
        self._training_manager = TrainingManager(self)
        self._self_play_manager = SelfPlayManager(self)
        self._ratings_managers: Dict[RatingTag, RatingsManager] = {}
        self._gpu_contention_manager = GpuContentionManager(self)
        self._remote_logging_manager = RemoteLoggingManager(self)

        # This line allows us to generate a per-thread stack trace by externally running:
        #
        # kill -s SIGUSR1 <pid>
        #
        # This is useful for diagnosing deadlocks.
        faulthandler.register(signal.SIGUSR1, all_threads=True)

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
    def default_training_gpu_id(self) -> GpuId:
        return self._default_training_gpu_id

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
    def build_params(self) -> BuildParams:
        return self._build_params

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

    def latest_gen(self) -> Generation:
        return self._training_manager.latest_gen()

    def get_gpu_lock_table_for_training(self) -> GpuContentionTable:
        return self._gpu_contention_manager.get_gpu_lock_table_for_training()

    def get_gpu_lock_table(self, gpu_id: GpuId) -> GpuContentionTable:
        return self._gpu_contention_manager.get_gpu_lock_table(gpu_id)

    def reset_self_play_locks(self):
        self._gpu_contention_manager.reset_self_play_locks()

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
            self._get_ratings_manager(conn.aux['tag']).add_server(conn)
        elif client_role == ClientRole.RATINGS_WORKER:
            self._get_ratings_manager(conn.aux['tag']).add_worker(conn)
        else:
            raise Exception(f'Unknown client type: {client_role}')

    def launch_recv_loop(self, msg_handler: MsgHandler, conn: ClientConnection, thread_name: str,
                         disconnect_handler: Optional[DisconnectHandler] = None,
                         preamble: Optional[Callable[[], None]] = None):
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
                         args=(msg_handler, disconnect_handler, conn, thread_name, preamble),
                         daemon=True).start()

    def handle_new_self_play_positions(self, n_augmented_positions: int):
        self._training_manager.handle_new_self_play_positions(n_augmented_positions)

    def handle_new_model(self):
        self._self_play_manager.notify_of_new_model()
        for manager in self._ratings_managers.values():
            manager.notify_of_new_model()

    def handle_log_msg(self, msg: JsonDict, conn: ClientConnection):
        self._remote_logging_manager.handle_log_msg(msg, conn)

    def handle_worker_exit(self, msg: JsonDict, conn: ClientConnection):
        self._remote_logging_manager.close_log_file(msg, conn.client_id)

    def broadcast_weights(self, conn: ClientConnection, gen: Generation):
        logger.debug(f'Broadcasting weights (gen={gen}) to {conn}')

        data = {
            'type': 'reload-weights',
            'generation': gen,
        }

        model_filename = self.organizer.get_model_filename(gen)
        with conn.socket.send_mutex():
            send_json(conn.socket.native_socket(), data)
            send_file(conn.socket.native_socket(), model_filename)

        logger.debug('Weights broadcast complete!')

    def set_ratings_priority(self, elevate: bool):
        self._gpu_contention_manager.set_ratings_priority(elevate)

    def _get_ratings_manager(self, tag: RatingTag) -> RatingsManager:
        if tag not in self._ratings_managers:
            self._ratings_managers[tag] = RatingsManager(self, tag)
        return self._ratings_managers[tag]

    def _launch_recv_loop_inner(
            self, msg_handler: MsgHandler, disconnect_handler: DisconnectHandler,
            conn: ClientConnection, thread_name: str, preamble: Optional[Callable[[], None]]):
        try:
            if preamble is not None:
                preamble()
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
                conn.active = False
                if disconnect_handler is not None:
                    disconnect_handler(conn)
                self._handle_disconnect(conn)
            except:
                logger.error(
                    f'Error handling disconnect in {thread_name} (conn={conn}):',
                    exc_info=True)
                self._shutdown_manager.request_shutdown(1)

    def _handle_disconnect(self, conn: ClientConnection):
        func = logger.debug if conn.client_role == ClientRole.RATINGS_WORKER else logger.info
        func(f'Handling disconnect: {conn}...')
        self._client_connection_manager.remove(conn)
        self._database_connection_manager.close_db_conns(threading.get_ident())
        self._remote_logging_manager.handle_disconnect(conn)
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
            self._self_play_manager.setup()
            self._training_manager.setup()
            self._client_connection_manager.start()

            if self._organizer.requires_retraining():
                self._training_manager.retrain_models()
                self._self_play_manager.signal_retraining_complete()
            else:
                self._self_play_manager.wait_for_gen0_completion()
                self._training_manager.train_gen1_model_if_necessary()

            if self._shutdown_manager.shutdown_requested():
                return
            while True:
                self._training_manager.wait_until_enough_training_data()
                self._training_manager.train_step()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)
