from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic import constants
from alphazero.logic.custom_types import ClientConnection, FileToTransfer
from alphazero.servers.loop_control.gpu_contention_table import Domain
from util.socket_util import JsonDict, SocketSendException

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import os
import subprocess
import threading
import time
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


class SelfPlayManager:

    @dataclass
    class ServerAux:
        """
        Auxiliary data stored per server connection.
        """
        launched: bool = False

    @dataclass
    class WorkerAux:
        """
        Auxiliary data stored per worker connection.
        """
        cond: threading.Condition = field(default_factory=threading.Condition)
        pending_pause_ack: bool = False
        pending_unpause_ack: bool = False
        gen: int = -1

    @dataclass
    class CommitInfo:
        """
        Houses per-worker metadata about the current generation of self-play data.

        Initially, the data is in an "unstaged" state - this means that if the worker process dies,
        the data is discarded.

        Once the data is fully received, it is "staged" - this means that the data is ready to be
        committed to the database. At this point, even if the worker process dies, the data will be
        committed to the database.

        Once the data is committed to the database, this object is removed from the
        SelfPlayManager's self._commit_info dict.

        This concept of staging/unstaging allows us to be flexible about the exact sequence of
        events that occur when a worker process dies (which typically happens for gen-0 and for
        periodic restarts).
        """
        n_rows: int = 0
        n_games: int = 0
        timestamp: int = 0
        metrics: Dict[str, int] = field(default_factory=dict)

        staged: bool = False  # see docstring for explanation

    def __init__(self, controller: LoopController):
        self._controller = controller

        self._scratch_dir = '/home/devuser/scratch/self-play-data'

        self._ready_server_conns: List[ClientConnection] = []
        self._worker_conns: List[ClientConnection] = []
        self._conns_lock = threading.Lock()
        self._conns_cond = threading.Condition(self._conns_lock)

        self._commit_info = defaultdict(SelfPlayManager.CommitInfo)
        self._n_committed_rows = 0
        self._commit_lock = threading.Lock()
        self._commit_cond = threading.Condition(self._commit_lock)

    def setup(self):
        os.makedirs(self._scratch_dir, exist_ok=True)

        with self._commit_lock:
            self._n_committed_rows = self._fetch_num_rows_in_db()

    def get_num_committed_rows(self):
        return self._n_committed_rows

    def run_gen0_if_necessary(self):
        num_rows = self._n_committed_rows
        if num_rows > 0:
            return

        with self._conns_lock:
            self._conns_cond.wait_for(lambda: self._ready_server_conns)
            conn = self._ready_server_conns[0]

        checkpoint = self._controller.get_next_checkpoint()
        self._launch_gen0_self_play(conn, checkpoint)
        self._collect_and_process_game_data()
        self._stop_gen0_self_play(conn)

    def run_until_checkpoint(self):
        self._controller._training_manager._set_checkpoint()
        checkpoint = self._controller.get_next_checkpoint()

        num_rows = self._n_committed_rows
        if checkpoint <= num_rows:
            return

        logger.info('Waiting for more training data... (current=%s, needed=%s)',
                    num_rows, checkpoint)

        self._launch_unlaunched_workers()
        self._controller.unhijack_all_self_play_tables()
        self._collect_and_process_game_data()

    def add_server(self, conn: ClientConnection):
        conn.aux = SelfPlayManager.ServerAux()
        self._controller.send_handshake_ack(conn)

        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'self-play-server',
            disconnect_handler=self._handle_server_disconnect)

    def add_worker(self, conn: ClientConnection):
        conn.aux = SelfPlayManager.WorkerAux()

        with self._conns_lock:
            self._worker_conns.append(conn)

        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'self-play-worker',
            disconnect_handler=self._handle_worker_disconnect)

    def _get_num_potential_rows(self):
        """
        Returns the number of rows there will be if all pending data is committed.

        Assumes that self._commit_lock is held.
        """
        return self._n_committed_rows + sum(info.n_rows for info in self._commit_info.values())

    def _fetch_num_rows_in_db(self) -> int:
        with self._controller.self_play_db_conn_pool.db_lock:
            # Return cumulative_positions for the last row of the self_play_data table:
            cursor = self._controller.self_play_db_conn_pool.get_cursor()
            cursor.execute("""SELECT cumulative_positions FROM self_play_data
                           ORDER BY gen DESC LIMIT 1""")
            row = cursor.fetchone()
            cursor.close()
        if row is not None:
            return row[0]

        return 0

    def _handle_server_disconnect(self, conn: ClientConnection):
        self._controller.stop_log_sync(conn)

    def _handle_worker_disconnect(self, conn: ClientConnection):
        logger.debug('Handling disconnect of %s...', conn)
        with self._conns_lock:
            self._worker_conns = [c for c in self._worker_conns if c.client_id != conn.client_id]

        with self._commit_lock:
            info = self._commit_info.get(conn.client_id, None)
            if info is not None and not info.staged:
                del self._commit_info[conn.client_id]

        aux: SelfPlayManager.WorkerAux = conn.aux
        with aux.cond:
            aux.pending_pause_ack = False
            aux.pending_unpause_ack = False
            aux.cond.notify_all()

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

    def _server_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('self-play-server received json message: %s', msg)

        if msg_type == 'ready':
            self._handle_ready(conn)
        elif msg_type == 'log-sync-start':
            self._controller.start_log_sync(conn, msg['log_filename'])
        elif msg_type == 'log-sync-stop':
            self._controller.stop_log_sync(conn, msg['log_filename'])
        elif msg_type == 'file-request':
            self._handle_file_request(conn, msg['files'])
        else:
            logger.warning('self-play-server: unknown message type: %s', msg)
        return False

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('self-play-worker received json message: %s', msg)

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'worker-ready':
            self._handle_worker_ready(conn)
        elif msg_type == 'heartbeat':
            self._handle_heartbeat(msg, conn)
        elif msg_type == 'self-play-data':
            self._handle_self_play_data(msg, conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('self-play-worker: unknown message type: %s', msg)
        return False

    def _launch_gen0_self_play(self, conn: ClientConnection, num_rows: int):
        logger.info('Requesting %s to perform gen-0 self-play...', conn)
        binary = self._construct_binary()

        data = {
            'type': 'start-gen0',
            'max_rows': num_rows,
            'binary': binary.to_dict(),
        }

        conn.socket.send_json(data)

    def _construct_binary(self) -> FileToTransfer:
        game = self._controller.run_params.game
        binary = FileToTransfer.from_src_scratch_path(
            source_path=self._controller.organizer_binary_path,
            scratch_path=f'target/bin/{game}',
            asset_path_mode='hash'
        )
        return binary

    def _stop_gen0_self_play(self, conn: ClientConnection):
        logger.info('Requesting %s to stop gen-0 self-play...', conn)

        data = {
            'type': 'stop-gen0',
        }

        conn.socket.send_json(data)

    def _launch_self_play(self, conn: ClientConnection):
        binary = self._construct_binary()
        data = {
            'type': 'start',
            'binary': binary.to_dict()
        }

        logger.info('Requesting %s to launch self-play...', conn)
        conn.socket.send_json(data)

        aux: SelfPlayManager.ServerAux = conn.aux
        aux.launched = True
        thread = threading.Thread(target=self._launch_self_play_restart_loop, args=(conn,),
                                  daemon=True, name=f'self-play-restart-loop')
        thread.start()

    def _launch_self_play_restart_loop(self, conn: ClientConnection):
        """
        There is currently a memory-leak in the c++ process. I suspect it comes from torchlib,
        although it's possible that the culprit lies in our code. The leak appears to contribute
        about 1GB of memory per hour. To mitigate this, we restart the process every hour.
        """
        try:
            while conn.active:
                time.sleep(3600)  # 1 hour
                self._restart(conn)
        except SocketSendException:
            logger.warning('Error sending to %s - worker likely disconnected', conn)
        except:
            logger.error('Unexpected error managing restart loop for %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _restart(self, conn: ClientConnection):
        logger.info('Restarting self-play for %s...', conn)
        binary = self._construct_binary()
        data = {
            'type': 'restart',
            'binary': binary.to_dict()
        }
        conn.socket.send_json(data)

    def _handle_ready(self, conn: ClientConnection):
        with self._conns_lock:
            self._ready_server_conns.append(conn)
            self._conns_cond.notify_all()

            if self._n_committed_rows > 0:
                self._launch_self_play(conn)

    def _manage_worker(self, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id

            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            table.activate(domain)
            self._pause(conn)

            while table.active(domain):
                if not table.acquire_lock(domain):
                    break
                self._refresh_weights_if_needed(conn)
                self._unpause(conn)
                if table.wait_for_lock_expiry(domain):
                    self._pause(conn)
                    table.release_lock(domain)
        except SocketSendException:
            logger.warning('Error sending to %s - worker likely disconnected', conn)
        except:
            logger.error('Unexpected error managing %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _pause(self, conn: ClientConnection):
        logger.debug('Pausing %s...', conn)

        aux: SelfPlayManager.WorkerAux = conn.aux
        aux.pending_pause_ack = True

        conn.socket.send_json({ 'type': 'pause' })

        with aux.cond:
            aux.cond.wait_for(lambda: not aux.pending_pause_ack)

        logger.debug('Pause of %s complete!', conn)

    def _unpause(self, conn: ClientConnection):
        logger.debug('Unpausing %s...', conn)

        aux: SelfPlayManager.WorkerAux = conn.aux
        aux.pending_unpause_ack = True

        conn.socket.send_json({ 'type': 'unpause' })

        with aux.cond:
            aux.cond.wait_for(lambda: not aux.pending_unpause_ack)

        logger.debug('Unpause of %s complete!', conn)

    def _handle_pause_ack(self, conn: ClientConnection):
        aux: SelfPlayManager.WorkerAux = conn.aux
        with aux.cond:
            aux.pending_pause_ack = False
            aux.cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        aux: SelfPlayManager.WorkerAux = conn.aux
        with aux.cond:
            aux.pending_unpause_ack = False
            aux.cond.notify_all()

    def _refresh_weights_if_needed(self, conn: ClientConnection):
        gen = self._controller.latest_gen()
        aux: SelfPlayManager.WorkerAux = conn.aux
        if aux.gen != gen:
            self._controller.broadcast_weights(conn, gen)
            aux.gen = gen

    def _handle_heartbeat(self, msg: JsonDict, conn: ClientConnection):
        rows = msg['rows']

        with self._commit_lock:
            self._commit_info[conn.client_id].n_rows = rows

            if self._get_num_potential_rows() >= self._controller.get_next_checkpoint():
                self._commit_cond.notify_all()

    def _launch_unlaunched_workers(self):
        with self._conns_lock:
            for conn in self._ready_server_conns:
                aux: SelfPlayManager.ServerAux = conn.aux
                if not aux.launched:
                    self._launch_self_play(conn)

    def _wait_until_checkpoint_reached(self):
        checkpoint = self._controller.get_next_checkpoint()
        with self._commit_lock:
            self._commit_cond.wait_for(lambda: self._get_num_potential_rows() >= checkpoint)

    def _collect_and_process_game_data(self):
        self._wait_until_checkpoint_reached()
        self._controller.get_gpu_lock_table_for_training().pre_acquire_lock(Domain.TRAINING)
        self._controller.hijack_all_self_play_tables()

        self._request_all_game_data()
        self._receive_all_game_data()
        self._update_self_play_db()

    def _request_all_game_data(self):
        n_rows_needed = self._controller.get_next_checkpoint() - self._n_committed_rows
        logger.debug('Requesting %s rows of data from all self-play workers...', n_rows_needed)

        n_rows_requested = 0

        with self._conns_lock:
            for conn in self._worker_conns:
                with self._commit_lock:
                    rows = self._commit_info[conn.client_id].n_rows
                n_rows_to_request = min(n_rows_needed - n_rows_requested, rows)
                if n_rows_to_request > 0:
                    self._request_game_data(conn, n_rows_to_request)
                n_rows_requested += n_rows_to_request

    def _request_game_data(self, conn: ClientConnection, n_rows: int):
        logger.debug('Requesting %s to send %s rows of data...', conn, n_rows)
        data = {
            'type': 'data-request',
            'n_rows': n_rows,
        }
        conn.socket.send_json(data)

    def _handle_self_play_data(self, msg: JsonDict, conn: ClientConnection):
        logger.debug('Handling self play data for %s...', conn)

        timestamp = msg['timestamp']
        n_games = msg['n_games']
        n_rows = msg['n_rows']
        metrics = msg.get('metrics', None)
        no_file = msg.get('no_file', False)

        # the JSON msg is immediately followed by the game data file. We receive it and save it to
        # scratch dir, to be merged later.
        game_filename = os.path.join(self._scratch_dir, f'{conn.client_id}.data')
        if no_file:
            # file should have been written by the c++ process
            assert os.path.isfile(game_filename), game_filename
        else:
            conn.socket.recv_file(game_filename)

        with self._commit_lock:
            info = self._commit_info[conn.client_id]

            assert n_rows <= info.n_rows, (n_rows, info.n_rows)

            info.n_rows = n_rows
            info.n_games = n_games
            info.timestamp = timestamp

            if metrics is not None:
                for column in constants.PERF_STATS_COLUMNS:
                    info.metrics[column] = metrics[column]

            info.staged = True
            if self._commit_data_fully_staged():
                self._commit_cond.notify_all()

    def _receive_all_game_data(self):
        logger.debug('Receiving all game data...')
        with self._commit_lock:
            self._commit_cond.wait_for(
                lambda: self._commit_data_fully_staged(), timeout=30)

            if not self._commit_data_fully_staged():
                raise Exception('Timed out waiting for self-play data from clients')

            client_ids = list(self._commit_info.keys())

        logger.debug('Clients with game data: %s', client_ids)

        game_filenames = [os.path.join(self._scratch_dir, f'{c}.data') for c in client_ids]

        for game_filename in game_filenames:
            assert os.path.isfile(game_filename), game_filename

        gen = self._controller.latest_gen()
        output_filename = self._controller.organizer.get_self_play_data_filename(gen)

        if len(game_filenames) == 1:
            # Easy case: we can just move the file to the correct location
            game_filename = game_filenames[0]
            subprocess.run(['mv', game_filename, output_filename], check=True)
        elif len(game_filenames) == 0:
            raise Exception('No game data received')
        else:
            self._controller.merge_game_log_files(game_filenames, output_filename)

    def _update_self_play_db(self):
        logger.debug('Updating self play db...')
        gen = self._controller.latest_gen()
        output_filename = self._controller.organizer.get_self_play_data_filename(gen)

        file_size = os.path.getsize(output_filename)

        with self._commit_lock:
            cumulative_positions = self._get_num_potential_rows()
            positions = cumulative_positions - self._n_committed_rows
            commit_info = dict(self._commit_info)

        has_metrics = any(info.metrics for info in commit_info.values())
        if has_metrics:
            assert all(info.metrics for info in commit_info.values()), \
                'Some workers have no metrics, but others do. This should not happen.'

        metrics_columns = [
            'client_id',
            'gen',
            'report_timestamp',
        ] + constants.PERF_STATS_COLUMNS

        metrics_insert_list = None
        if has_metrics:
            metrics_insert_list = [(client_id, gen, info.timestamp,
                                    *[info.metrics[col] for col in constants.PERF_STATS_COLUMNS])
                                   for client_id, info in commit_info.items()]

        values_str = ', '.join(['?' for _ in metrics_columns])

        infos = list(commit_info.values())
        n_positions_evaluated = sum(info.metrics.get('positions_evaluated', 0) for info in infos)
        n_batches_evaluated = sum(info.metrics.get('batches_evaluated', 0) for info in infos)
        n_games = sum(info.n_games for info in commit_info.values())

        self_play_data_columns = [
            'gen',
            'positions',
            'cumulative_positions',
            'positions_evaluated',
            'batches_evaluated',
            'games',
            'file_size',
        ]

        self_play_data_insert_tuple = (
            gen,
            positions,
            cumulative_positions,
            n_positions_evaluated,
            n_batches_evaluated,
            n_games,
            file_size,
        )

        with self._controller.self_play_db_conn_pool.db_lock:
            db_conn = self._controller.self_play_db_conn_pool.get_connection()
            cursor = db_conn.cursor()

            if has_metrics:
                cursor.executemany(f"""INSERT INTO metrics ({', '.join(metrics_columns)})
                                    VALUES ({values_str})""", metrics_insert_list)

            cursor.execute(
                f"""INSERT INTO self_play_data ({', '.join(self_play_data_columns)})
                VALUES ({', '.join(['?' for _ in self_play_data_columns])})""",
                self_play_data_insert_tuple)

            cursor.close()
            db_conn.commit()

        with self._commit_lock:
            self._n_committed_rows = self._get_num_potential_rows()
            self._commit_info.clear()

        self._controller.handle_new_self_play_data(gen, positions, file_size)

    def _commit_data_fully_staged(self):
        return all(info.staged for info in self._commit_info.values())

    def _handle_worker_ready(self, conn: ClientConnection):
        thread = threading.Thread(target=self._manage_worker, args=(conn, ),
                                  daemon=True, name=f'manage-self-play-worker')
        thread.start()

    def _handle_file_request(self, conn: ClientConnection, files: List[JsonDict]):
        self._controller.handle_file_request(conn, files)
