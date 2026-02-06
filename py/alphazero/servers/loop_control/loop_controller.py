from .base_dir import Benchmark, Scratch, Workspace
from .self_eval_manager import SelfEvalManager
from .client_connection_manager import ClientConnectionManager
from .database_connection_manager import DatabaseConnectionManager
from .directory_organizer import DirectoryOrganizer
from .eval_vs_benchmark_manager import EvalVsBenchmarkManager
from .gpu_contention_manager import GpuContentionManager
from .gpu_contention_table import GpuContentionTable
from .log_syncer import LogSyncer
from .output_dir_syncer import OutputDirSyncer
from .params import LoopControllerParams
from .ratings_manager import RatingsManager
from .self_play_manager import SelfPlayManager
from .training_manager import TrainingManager

from alphazero.logic import constants
from alphazero.logic.benchmark_record import BenchmarkData
from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientConnection, ClientRole, DisconnectHandler, Domain, \
     Generation, GpuId, MsgHandler, RatingTag, ShutdownAction
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.logic.signaling import register_standard_server_signals
from games.game_spec import GameSpec
from games.index import get_game_spec
from shared.basic_types import SearchParadigm
from shared.rating_params import RatingParams
from shared.training_params import TrainingParams
from util.py_util import atomic_cp, sha256sum
from util.socket_util import JsonDict, SocketRecvException, SocketSendException, send_file, \
    send_json
from util.sqlite3_util import DatabaseConnectionPool
from util import ssh_util

import logging
import os
import shutil
import socket
import threading
from typing import Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class LoopController:
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
                 run_params: RunParams, build_params: BuildParams, rating_params: RatingParams):
        self._game_spec = get_game_spec(run_params.game)
        self._params = params
        self._training_params = training_params
        self._run_params = run_params
        self._build_params = build_params
        self.rating_params = rating_params
        self._default_training_gpu_id = GpuId(constants.LOCALHOST_IP, params.cuda_device)
        self._socket: Optional[socket.socket] = None

        # On cloud setups like runpod.io or GCP, we typically have access to two filesystems:
        #
        # 1. A fast local filesystem, which is wiped after each session.
        # 2. A slow persistent filesystem, whose contents persist across sessions.
        #
        # On such setups, we work with the local filesystem, assumed to be available at
        # ~/scratch/, and periodically sync to the persistent filesystem, assumed to be available
        # at /workspace/.
        #
        # On other setups, we solely work on /workspace/, which is assumed to be the local
        # filesystem.
        #
        # For now, we detect whether we are on a cloud setup (referred to as an "ephemeral local
        # disk env") by checking if ~/scratch and /workspace are on the same filesystem. If we
        # encounter environments where this does not work as intended, we can rethink this.
        scratch_fs = os.stat('/home/devuser/scratch').st_dev
        workspace_fs = os.stat('/workspace').st_dev
        self._on_ephemeral_local_disk_env = (scratch_fs != workspace_fs) or params.simulate_cloud

        if self._on_ephemeral_local_disk_env:
            self._organizer = DirectoryOrganizer(run_params, base_dir_root=Scratch)
            self._persistent_organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
        else:
            self._organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
            self._persistent_organizer = None

        self._shutdown_manager = ShutdownManager()
        self._client_connection_manager = ClientConnectionManager(self)
        self._database_connection_manager = DatabaseConnectionManager(self)
        self._log_syncer = LogSyncer(self)
        self._training_manager = TrainingManager(self)
        self._self_play_manager = SelfPlayManager(self)
        self._eval_managers: Dict[RatingTag, EvalVsBenchmarkManager] = {}
        self._ratings_managers: Dict[RatingTag, RatingsManager] = {}
        self._self_eval_manager: Optional[SelfEvalManager] = None
        self._gpu_contention_manager = GpuContentionManager(self)

        # OutputDirSyncer must be the LAST constructed sub-manager, to ensure proper shutdown
        # sequencing
        self._output_dir_syncer = OutputDirSyncer(self)

        register_standard_server_signals(ignore_sigint=params.ignore_sigint)

    def run(self):
        """
        Entry-point into the LoopController.
        """
        try:
            self._setup()
            self._start_threads()
            self._shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('loop_controller caught Ctrl-C')
            self._shutdown_manager.request_shutdown(1)
        except SystemExit:
            logger.info('loop_controller caught SystemExit')
            self._shutdown_manager.request_shutdown(1)
        except Exception as e:
            logger.error('Unexpected error in run(): %s', e, exc_info=True)
            self._shutdown_manager.request_shutdown(1)
        finally:
            self._shutdown_manager.shutdown()

    @property
    def search_paradigm(self) -> SearchParadigm:
        return self.game_spec.model_configs[self.params.model_cfg].search_paradigm

    @property
    def on_ephemeral_local_disk_env(self) -> bool:
        return self._on_ephemeral_local_disk_env

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
    def persistent_organizer(self) -> DirectoryOrganizer:
        assert self._persistent_organizer is not None
        return self._persistent_organizer

    @property
    def organizer_binary_path(self) -> str:
        if self._on_ephemeral_local_disk_env:
            return self._persistent_organizer.binary_filename
        else:
            return self._organizer.binary_filename

    @property
    def params(self) -> LoopControllerParams:
        return self._params

    @property
    def training_params(self) -> TrainingParams:
        return self._training_params

    @property
    def run_params(self) -> RunParams:
        return self._run_params

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

    def db_conn_pools(self) -> List[DatabaseConnectionPool]:
        return self._database_connection_manager.pools()

    def latest_gen(self) -> Generation:
        return self._training_manager.latest_gen()

    def get_gpu_lock_table_for_training(self) -> GpuContentionTable:
        return self._gpu_contention_manager.get_gpu_lock_table_for_training()

    def get_other_gpu_lock_tables(self, table: GpuContentionTable) -> List[GpuContentionTable]:
        """
        Returns all GPU lock tables on the same machine as the given table, asides from the
        table itself.
        """
        return self._gpu_contention_manager.get_other_gpu_lock_tables(table)

    def get_gpu_lock_table(self, gpu_id: GpuId) -> GpuContentionTable:
        return self._gpu_contention_manager.get_gpu_lock_table(gpu_id)

    def register_shutdown_action(self, action: ShutdownAction, descr: str):
        self._shutdown_manager.register(action, descr)

    def request_shutdown(self, return_code: int):
        self._shutdown_manager.request_shutdown(return_code)

    def handle_new_client_connection(self, conn: ClientConnection):
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
            self._get_ratings_manager(conn.rating_tag).add_server(conn)
        elif client_role == ClientRole.RATINGS_WORKER:
            self._get_ratings_manager(conn.rating_tag).add_worker(conn)
        elif client_role == ClientRole.EVAL_VS_BENCHMARK_SERVER:
            self._get_eval_manager(conn.rating_tag).add_server(conn)
        elif client_role == ClientRole.EVAL_VS_BENCHMARK_WORKER:
            self._get_eval_manager(conn.rating_tag).add_worker(conn)
        elif client_role == ClientRole.SELF_EVAL_SERVER:
            self._get_self_eval_manager().add_server(conn)
        elif client_role == ClientRole.SELF_EVAL_WORKER:
            self._get_self_eval_manager().add_worker(conn)
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

    def handle_new_model(self):
        # this hijack + unhijack causes all self-play workers to pause and then unpause, which
        # triggers a weights-refresh. See SelfPlayManager._manage_worker().
        self._gpu_contention_manager.hijack_all_self_play_tables()
        self._gpu_contention_manager.unhijack_all_self_play_tables()

        for tag, manager in self._ratings_managers.items():
            assert tag is not None  # defensive programming, this indicates a bug
            manager.notify_of_new_model()

        for tag, manager in self._eval_managers.items():
            assert tag is not None  # defensive programming, this indicates a bug
            manager.notify_of_new_model()

        if self._self_eval_manager is not None:
            self._self_eval_manager.notify_of_new_model()

    def handle_new_self_play_data(self, gen: Generation, n_rows: int, file_size: int):
        self._training_manager.notify_of_new_self_play_data(gen, n_rows, file_size)

    def broadcast_weights(self, conn: ClientConnection, gen: Generation):
        logger.debug('Broadcasting weights to %s', conn)

        required_rows = self.get_checkpoint() - self.get_num_committed_rows()

        data1 = {
            'type': 'data-pre-request',
            'n_rows_limit': required_rows,
        }

        data2 = {
            'type': 'reload-weights',
        }

        model_filename = self.organizer.get_model_filename(gen)
        with conn.socket.send_mutex():
            send_json(conn.socket.native_socket(), data1)
            send_json(conn.socket.native_socket(), data2)
            send_file(conn.socket.native_socket(), model_filename)

        logger.debug('Weights broadcast complete!')

    def send_handshake_ack(self, conn: ClientConnection):
        ssh_pub_key = ssh_util.get_pub_key()
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self.game_spec.name,
            'tag': self.run_params.tag,
            'ssh_pub_key': ssh_pub_key,
            'on_ephemeral_local_disk_env': self.on_ephemeral_local_disk_env,
        }
        conn.socket.send_json(reply)

    def handle_file_request(self, conn: ClientConnection, files: List[JsonDict]):
        logger.debug('Handling file request from %s: %s', conn, files)
        conn.socket.send_json({
            'type': 'file-transfer',
            'files': files,
        })
        for f in files:
            conn.socket.send_file(f['source_path'])

    def set_domain_priority(self, domain: Domain, elevate: bool):
        self._gpu_contention_manager.set_domain_priority(domain, elevate)

    def get_num_committed_rows(self):
        return self._self_play_manager.get_num_committed_rows()

    def get_checkpoint(self):
        return self._training_manager.get_checkpoint()

    def estimate_upcoming_checkpoint(self):
        return self._training_manager.estimate_upcoming_checkpoint()

    def start_log_sync(self, conn: ClientConnection, remote_filename: str):
        self._log_syncer.register(conn, remote_filename)

    def stop_log_sync(self, conn: ClientConnection, remote_filename: Optional[str] = None):
        self._log_syncer.unregister(conn, remote_filename)

    def spawn_log_sync_thread(self):
        self._log_syncer.spawn_sync_thread()

    def wait_for_log_sync_thread(self):
        self._log_syncer.wait_for_sync_thread()

    def merge_game_log_files(self, input_filenames: List[str], output_filename: str):
        self._training_manager.merge_game_log_files(input_filenames, output_filename)

    def _get_ratings_manager(self, tag: RatingTag) -> RatingsManager:
        if tag not in self._ratings_managers:
            self._ratings_managers[tag] = RatingsManager(self, tag)
        return self._ratings_managers[tag]

    def _get_eval_manager(self, tag: RatingTag) -> EvalVsBenchmarkManager:
        if tag not in self._eval_managers:
            benchmark_data = BenchmarkData(self.game_spec.name, self.params.benchmark_tag)
            benchmark_tag = benchmark_data.setup_rundir()
            self._copy_eval_db(benchmark_tag)
            self._eval_managers[tag] = EvalVsBenchmarkManager(self, benchmark_tag, tag)
        return self._eval_managers[tag]

    def _copy_eval_db(self, benchmark_tag: str):
        eval_db_file = self.organizer.eval_db_filename(benchmark_tag)
        db = RatingDB(eval_db_file)
        if os.path.exists(db.db_filename) and not db.is_empty():
            logger.debug(f"{eval_db_file} already exists and is not empty; skip copying.")
            return

        run_params = RunParams(self.game_spec.name, benchmark_tag)
        benchmark_organizer = DirectoryOrganizer(run_params, base_dir_root=Benchmark)
        shutil.copyfile(benchmark_organizer.benchmark_db_filename, eval_db_file)
        logger.debug(f"copy db from: {benchmark_organizer.benchmark_db_filename} "
                     f"to: {eval_db_file}")

    def _get_self_eval_manager(self) -> SelfEvalManager:
        if not self._self_eval_manager:
            self._self_eval_manager = SelfEvalManager(self)
        return self._self_eval_manager

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
            if self._shutdown_manager.shutdown_requested():
                pass
            logger.warning(
                'Encountered SocketRecvException in %s (conn=%s):', thread_name, conn)
        except SocketSendException:
            logger.warning(
                'Encountered SocketSendException in %s (conn=%s):', thread_name, conn,
                exc_info=True, stack_info=True, stacklevel=10)
        except:
            logger.error(
                'Unexpected error in %s (conn=%s):', thread_name, conn, exc_info=True)
            self._shutdown_manager.request_shutdown(1)
        finally:
            try:
                conn.active = False
                if disconnect_handler is not None:
                    disconnect_handler(conn)
                self._handle_disconnect(conn)
            except:
                logger.error(
                    'Error handling disconnect in %s (conn=%s):', thread_name, conn,
                    exc_info=True)
                self._shutdown_manager.request_shutdown(1)

    def _handle_disconnect(self, conn: ClientConnection):
        log_level = ClientRole.connection_log_level(conn.client_role)
        logger.log(log_level, 'Handling disconnect: %s...', conn)
        self._client_connection_manager.remove(conn)
        self._database_connection_manager.close_db_conns(threading.get_ident())
        self._log_syncer.unregister(conn)
        conn.socket.close()

    def _init_socket(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setblocking(True)
        self._socket.bind(('0.0.0.0', self.params.port))
        self._socket.listen()

        self.register_shutdown_action(lambda: self._socket.close(), 'socket-close')

    def _setup_output_dir(self):
        self._organizer.dir_setup(paradigm=self.search_paradigm.value)
        if self._on_ephemeral_local_disk_env:
            if not self._restore_prior_run():
                self._persistent_organizer.dir_setup(paradigm=self.search_paradigm.value)

    def _restore_prior_run(self):
        assert self._on_ephemeral_local_disk_env

        if not os.path.isdir(self._persistent_organizer.base_dir):
            return False

        logger.info('Restoring prior run from %s...', self._persistent_organizer.base_dir)

        # First, copy database files
        logger.info('Copying database files...')
        shutil.copytree(self._persistent_organizer.databases_dir, self._organizer.databases_dir,
                        dirs_exist_ok=True)

        # Next, copy all model files, as they are required for ratings runs
        logger.info('Copying model files...')
        shutil.copytree(self._persistent_organizer.models_dir, self._organizer.models_dir,
                        dirs_exist_ok=True)

        # Next, copy self-play-data. We only need to copy generations that might be used for
        # future training epochs.
        last_gen = self._persistent_organizer.get_latest_self_play_generation(default=0)
        gen = self._training_manager.get_oldest_required_gen()
        logger.info(f'Copying self-play data from gen {gen} to {last_gen}...')
        while gen <= last_gen:
            src = self._persistent_organizer.get_self_play_data_filename(gen)
            if src is not None and os.path.isfile(src):
                dst = self._organizer.get_self_play_data_filename(gen)
                shutil.copy(src, dst)
            gen += 1

        # Copy the most recent checkpoint
        checkpoint_gen = self._persistent_organizer.get_last_checkpointed_generation()
        if checkpoint_gen is not None:
            checkpoint_filename = self._persistent_organizer.get_checkpoint_filename(checkpoint_gen)
            shutil.copy(checkpoint_filename, self._organizer.checkpoints_dir)

        logger.info('Prior run restoration complete!')
        return True

    def _copy_binary_file(self):
        if self._on_ephemeral_local_disk_env:
            organizer = self._persistent_organizer
        else:
            organizer = self._organizer

        binary_path = self.build_params.get_binary_path(self.game_spec.name)
        target_file = organizer.binary_filename
        if os.path.exists(target_file):
            hash = sha256sum(target_file)
            if hash == sha256sum(binary_path):
                logger.debug('Binary file already exists, skipping copy')
                return
            elif not self.build_params.override_binary:
                message = f"""Hash mismatch for binary file {target_file} and {binary_path}.
                Include --override-binary to override or have the matching binary: {binary_path}"""
                raise Exception(message)
            else:
                message = f"""Hash mismatch for binary file {target_file} and {binary_path}.
                Overriding binary file with the one in {binary_path}"""
                logger.warning(message)
        else:
            logger.info('Copying binary file to persistent run dir...')

        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        atomic_cp(binary_path, target_file)

    def _get_binary_path(self, benchmark_organizer: Optional[DirectoryOrganizer] = None) -> str:
        use_stored_binary = self.build_params.use_stored_binary
        if not use_stored_binary:
            return self.build_params.get_binary_path(self.game_spec.name)
        else:
            if benchmark_organizer is None:
                stored_binary_path = self.organizer_binary_path
            else:
                stored_binary_path = benchmark_organizer.binary_filename

            if os.path.exists(stored_binary_path):
                return stored_binary_path
            else:
                raise Exception(f'Stored binary path does not exist: {stored_binary_path}')

    def _training_loop(self):
        try:
            self._self_play_manager.setup()
            self._training_manager.setup()

            if self._organizer.requires_retraining():
                self._training_manager.retrain_models()

            self._self_play_manager.run_gen0_if_necessary()
            self._training_manager.train_gen1_model_if_necessary()

            if self._shutdown_manager.shutdown_requested():
                return

            while True:
                self._self_play_manager.run_until_checkpoint()
                self._training_manager.train_step()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _task_loop(self):
        """
        task loop is used for performing tasks such as evaluation, ratings and benchmarking without
        running self-play or training.
        """
        try:
            self._setup()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _setup(self):
        logger.info('Performing LoopController setup...')
        self._setup_output_dir()
        self._organizer.acquire_lock(self._shutdown_manager.register)
        if self.build_params.binary_path is None and not self.build_params.debug_build:
            self._copy_binary_file()
        self._init_socket()

    def _start_threads(self):
        self._output_dir_syncer.start()
        self._client_connection_manager.start()

        if not self.params.task_mode:
            self._organizer.assert_not_frozen()
            logger.info('Starting self-play and training threads...')
            threading.Thread(target=self._training_loop, name='main_loop', daemon=True).start()
