from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.custom_types import ClientConnection
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketSendException
from util import ssh_util

from dataclasses import dataclass, field
import os
import threading
import time
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = get_logger()


class SelfPlayManager:

    @dataclass
    class CommitInfo:
        pending_metrics: list = field(default_factory=list)
        client_ids_with_pending_game_data: set = field(default_factory=set)
        client_ids_with_game_data: list = field(default_factory=list)
        n_committed_rows: int = 0
        n_pending_rows: int = 0
        n_pending_games: int = 0
        pending_runtime: int = 0

    def __init__(self, controller: LoopController):
        self._controller = controller

        self._scratch_dir = '/home/devuser/scratch/self-play-data'

        self._gen0_complete = False
        self._gen0_lock = threading.Lock()
        self._gen0_cond = threading.Condition(self._gen0_lock)

        self._ready_conns: List[ClientConnection] = []
        self._ready_lock = threading.Lock()
        self._ready_cond = threading.Condition(self._ready_lock)

        self._commit_info = SelfPlayManager.CommitInfo()
        self._commit_info_lock = threading.Lock()
        self._commit_info_cond = threading.Condition(self._commit_info_lock)

    def setup(self):
        os.makedirs(self._scratch_dir, exist_ok=True)

        with self._commit_info_lock:
            self._commit_info.n_committed_rows = self._fetch_num_rows_in_db()

    def get_num_rows(self):
        with self._commit_info_lock:
            return self._get_num_rows()

    def run_gen0_if_necessary(self):
        additional_gen0_rows_needed = self._num_additional_gen0_positions_needed()
        if additional_gen0_rows_needed == 0:
            self._gen0_complete = True
            return

        with self._ready_lock:
            self._ready_cond.wait_for(lambda: self._ready_conns)
            conn = self._ready_conns[0]

        self._launch_gen0_self_play(conn, additional_gen0_rows_needed)
        self._wait_until_checkpoint_reached()

        with self._gen0_cond:
            self._gen0_cond.wait_for(lambda: self._gen0_complete)

        self._collect_and_process_game_data()

    def run_until_checkpoint(self):
        checkpoint = self._controller.get_next_checkpoint()

        num_rows = self.get_num_rows()
        if checkpoint <= num_rows:
            return

        logger.info('Waiting for more training data... (current=%s, needed=%s)',
                    num_rows, checkpoint)

        self._launch_unlaunched_workers()
        self._controller.unhijack_all_self_play_tables()
        self._wait_until_checkpoint_reached()
        self._collect_and_process_game_data()
        self._controller.hijack_all_self_play_tables()

    def add_server(self, conn: ClientConnection):
        ssh_pub_key = ssh_util.get_pub_key()
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self._controller.game_spec.name,
            'tag': self._controller.run_params.tag,
            'ssh_pub_key': ssh_pub_key,
            'on_ephemeral_local_disk_env': self._controller.on_ephemeral_local_disk_env,
            'asset-requirements': self._controller.get_asset_requirements(),
        }
        conn.socket.send_json(reply)

        assets_request = conn.socket.recv_json()
        assert assets_request['type'] == 'assets-request'
        for asset in assets_request['assets']:
            conn.socket.send_file(asset)

        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'self-play-server',
            disconnect_handler=self._handle_server_disconnect)

    def add_worker(self, conn: ClientConnection):
        conn.aux['ack_cond'] = threading.Condition()
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'self-play-worker',
            disconnect_handler=self._handle_worker_disconnect)

    def _get_num_rows(self):
        return self._commit_info.n_committed_rows + self._commit_info.n_pending_rows

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
        # If we really want, we could implement graceful handling for this case, but it's simpler
        # to just shut down the server.
        if not self._gen0_complete:
            raise Exception('Server disconnected before gen-0 self-play was complete')
        self._controller.stop_log_sync(conn)

    def _handle_worker_disconnect(self, conn: ClientConnection):
        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_pause_ack', None)
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

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
        elif msg_type == 'gen0-complete':
            self._handle_gen0_complete()
        else:
            logger.warning('self-play-server: unknown message type: %s', msg)
        return False

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'weights-request':
            self._handle_weights_request(conn)
        elif msg_type == 'heartbeat':
            self._handle_heartbeat(msg, conn)
        elif msg_type == 'self-play-data':
            self._handle_self_play_data(msg, conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('self-play-worker: unknown message type: %s', msg)
        return False

    def _set_gen0_completion(self, complete: bool, log=True):
        """
        Assumes self._gen0_lock is already acquired.
        """
        if self._gen0_complete:
            return

        self._gen0_complete = complete
        if complete and log:
            logger.info('Gen-0 self-play complete!')

    def _launch_gen0_self_play(self, conn: ClientConnection, num_rows: int):
        logger.info('Requesting %s to perform gen-0 self-play...', conn)

        data = {
            'type': 'start-gen0',
            'max_rows': num_rows,
        }

        conn.socket.send_json(data)

    def _launch_self_play(self, conn: ClientConnection):
        data = {
            'type': 'start',
        }

        logger.info('Requesting %s to launch self-play...', conn)
        conn.socket.send_json(data)

        conn.aux['launched'] = True
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
        data = {
            'type': 'restart',
        }
        conn.socket.send_json(data)

    def _handle_ready(self, conn: ClientConnection):
        with self._ready_lock:
            self._ready_conns.append(conn)
            self._ready_cond.notify_all()

            with self._gen0_lock:
                if not self._gen0_complete:
                    return

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
        data = {
            'type': 'pause',
        }
        conn.aux['pending_pause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_pause_ack' not in conn.aux)

        logger.debug('Pause of %s complete!', conn)

    def _unpause(self, conn: ClientConnection):
        logger.debug('Unpausing %s...', conn)
        data = {
            'type': 'unpause',
        }
        conn.aux['pending_unpause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_unpause_ack' not in conn.aux)

        logger.debug('Unpause of %s complete!', conn)

    def _handle_pause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            start_ts = conn.aux.get('start_ts', None)
            if start_ts is not None:
                elapsed = time.time_ns() - start_ts
                total_runtime = conn.aux.get('total_runtime', 0)
                conn.aux['total_runtime'] = total_runtime + elapsed
                del conn.aux['start_ts']
            conn.aux.pop('pending_pause_ack', None)
            cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            if 'start_ts' not in conn.aux:
                conn.aux['start_ts'] = time.time_ns()
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

    def _refresh_weights_if_needed(self, conn: ClientConnection):
        gen = self._controller.latest_gen()
        if conn.aux.get('gen', None) != gen:
            self._controller.broadcast_weights(conn, gen)
            conn.aux['gen'] = gen

    def _handle_gen0_complete(self):
        with self._gen0_cond:
            self._gen0_complete = True
            self._gen0_cond.notify_all()

    def _num_additional_gen0_positions_needed(self) -> int:
        total_needed = self._controller.training_params.samples_per_window()
        with self._controller.self_play_db_conn_pool.db_lock:
            cursor = self._controller.self_play_db_conn_pool.get_cursor()
            cursor.execute(
                'SELECT gen, cumulative_positions FROM self_play_data ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return total_needed
        gen, cumulative_positions = row
        if gen > 0:
            return 0

        assert gen == 0, gen
        return max(0, total_needed - cumulative_positions)

    def _handle_heartbeat(self, msg: JsonDict, conn: ClientConnection):
        rows = msg['rows']

        with self._commit_info_lock:
            prev_rows = conn.aux.get('rows', 0)
            conn.aux['rows'] = rows
            self._commit_info.n_pending_rows += rows - prev_rows

            if self._get_num_rows() >= self._controller.get_next_checkpoint():
                self._commit_info_cond.notify_all()

    def _launch_unlaunched_workers(self):
        with self._ready_lock:
            for conn in self._ready_conns:
                if not conn.aux.get('launched', False):
                    self._launch_self_play(conn)

    def _wait_until_checkpoint_reached(self):
        checkpoint = self._controller.get_next_checkpoint()
        with self._commit_info_lock:
            self._commit_info_cond.wait_for(lambda: self._get_num_rows() >= checkpoint)

    def _collect_and_process_game_data(self):
        self._request_all_game_data()
        self._receive_all_game_data()
        self._update_self_play_db()
        self._reset_row_data()

    def _request_all_game_data(self):
        logger.debug('Requesting data from all self-play workers...')
        n_rows_needed = self._controller.get_next_checkpoint() - self._commit_info.n_committed_rows

        n_rows_requested = 0

        with self._ready_lock:
            for conn in self._ready_conns:
                rows = conn.aux.get('rows', 0)
                n_rows_to_request = min(n_rows_needed - n_rows_requested, rows)
                if n_rows_to_request > 0:
                    self._request_game_data(conn, n_rows_to_request)
                n_rows_requested += n_rows_to_request

    def _request_game_data(self, conn: ClientConnection, n_rows: int):
        with self._commit_info_lock:
            self._commit_info.client_ids_with_pending_game_data.add(conn.client_id)

        logger.debug('Requesting %s to send %s rows of data...', conn, n_rows)
        data = {
            'type': 'data-request',
            'n_rows': n_rows,
        }
        conn.socket.send_json(data)

    def _handle_self_play_data(self, msg: JsonDict, conn: ClientConnection):
        logger.debug('Handling self play data for %s...', conn)

        timestamp = msg['timestamp']
        gen = msg['gen']
        n_games = msg['n_games']
        metrics = msg.get('metrics', None)
        no_file = msg.get('no_file', False)

        cond = conn.aux['ack_cond']
        with cond:
            start_ts = conn.aux.get('start_ts', None)
            total_runtime = conn.aux.pop('total_runtime', 0)
            if start_ts is not None:
                now = time.time_ns()
                elapsed = now - start_ts
                total_runtime += elapsed
                conn.aux['start_ts'] = now

        # the JSON msg is immediately followed by the game data file. We receive it and save it to
        # scratch dir, to be merged later.
        game_filename = os.path.join(self._scratch_dir, f'{conn.client_id}.data')
        if no_file:
            # file should have been written by the c++ process
            assert os.path.isfile(game_filename), game_filename
        else:
            conn.socket.recv_file(game_filename)

        with self._commit_info_lock:
            self._commit_info.pending_runtime += total_runtime
            self._commit_info.n_pending_games += n_games
            if metrics is not None:
                self._commit_info.pending_metrics.append((conn.client_id, timestamp, metrics))
            self._commit_info.client_ids_with_pending_game_data.pop(conn.client_id)
            self._commit_info.client_ids_with_game_data.append(conn.client_id)
            if not self._commit_info.client_ids_with_pending_game_data:
                self._commit_info_cond.notify_all()

        if gen == 0:
            logger.info('Client %s has finished self-play', conn.client_id)
            conn.socket.send_json({'type': 'quit'})

    def _receive_all_game_data(self):
        logger.debug('Receiving all game data...')
        with self._commit_info_lock:
            self._commit_info_cond.wait_for(
                lambda: not self._commit_info.client_ids_with_pending_game_data, timeout=30)

            if self._commit_info.client_ids_with_pending_game_data:
                raise Exception('Timed out waiting for self-play data from clients')

            client_ids = list(self._commit_info.client_ids_with_game_data)

        game_filenames = [os.path.join(self._scratch_dir, f'{c}.data') for c in client_ids]

        for game_filename in game_filenames:
            assert os.path.isfile(game_filename), game_filename

        gen = self._controller.latest_gen()
        output_filename = self._controller.organizer.get_self_play_data_filename(gen)

        if len(game_filenames) == 1:
            # Easy case: we can just move the file to the correct location
            game_filename = game_filenames[0]
            os.rename(game_filename, output_filename)
        elif len(game_filenames) == 0:
            raise Exception('No game data received')
        else:
            # Harder case: we need to weave the files together, sorting by game-start-timestamp
            # This will require an FFI call to the c++ code
            raise NotImplementedError('Multiple self-play servers temporarily disabled')

    def _update_self_play_db(self):
        logger.debug('Updating self play db...')
        gen = self._controller.latest_gen()
        output_filename = self._controller.organizer.get_self_play_data_filename(gen)

        file_size = os.path.getsize(output_filename)

        with self._commit_info_lock:
            positions = self._commit_info.n_pending_rows
            cumulative_positions = self._commit_info.n_committed_rows + positions
            n_games = self._commit_info.n_pending_games
            metrics_list = list(self._commit_info.pending_metrics)

        metrics_columns = [
            'client_id',
            'gen',
            'report_timestamp',
            'cache_hits',
            'cache_misses',
            'positions_evaluated',
            'batches_evaluated',
            'full_batches_evaluated',
        ]
        values_str = ', '.join(['?' for _ in metrics_columns])

        metrics_insert_list = [
            (client_id,
             gen,
             timestamp,
             metrics['cache_hits'],
             metrics['cache_misses'],
             metrics['positions_evaluated'],
             metrics['batches_evaluated'],
             metrics['full_batches_evaluated'])
            for client_id, timestamp, metrics in metrics_list]

        positions_evaluated = sum(metrics['positions_evaluated'] for _, _, metrics in metrics_list)
        batches_evaluated = sum(metrics['batches_evaluated'] for _, _, metrics in metrics_list)
        runtime = sum(metrics['runtime'] for _, _, metrics in metrics_list)

        self_play_data_columns = [
            'gen',
            'positions',
            'cumulative_positions',
            'positions_evaluated',
            'batches_evaluated',
            'games',
            'runtime',
            'file_size',
        ]

        self_play_data_insert_tuple = (
            gen,
            positions,
            cumulative_positions,
            positions_evaluated,
            batches_evaluated,
            n_games,
            runtime,
            file_size,
        )

        with self._controller.self_play_db_conn_pool.db_lock:
            db_conn = self._controller.self_play_db_conn_pool.get_connection()
            cursor = db_conn.cursor()

            if metrics_insert_list:
                cursor.executemany(f"""INSERT INTO metrics ({', '.join(metrics_columns)})
                                    VALUES ({values_str})""", metrics_insert_list)

            cursor.execute(
                f"""INSERT INTO self_play_data ({', '.join(self_play_data_columns)})
                VALUES ({', '.join(['?' for _ in self_play_data_columns])})""",
                self_play_data_insert_tuple)

        self._controller.handle_new_self_play_data(gen, positions, file_size)

    def _reset_row_data(self):
        with self._commit_info_lock:
            self._commit_info.n_committed_rows += self._commit_info.n_pending_rows
            self._commit_info.n_pending_rows = 0
            self._commit_info.n_pending_games = 0
            self._commit_info.pending_runtime = 0
            self._commit_info.pending_metrics = []
            self._commit_info.client_ids_with_pending_game_data = set()
            self._commit_info.client_ids_with_game_data = []

    def _handle_weights_request(self, conn: ClientConnection):
        thread = threading.Thread(target=self._manage_worker, args=(conn,),
                                  daemon=True, name=f'manage-self-play-worker')
        thread.start()
