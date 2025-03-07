from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.custom_types import ClientConnection, ClientId, Generation
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketSendException
from util import ssh_util

from collections import defaultdict
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = get_logger()


class SelfPlayManager:
    def __init__(self, controller: LoopController):
        self._controller = controller

        self._gen0_complete = False
        self._gen0_lock = threading.Lock()
        self._gen0_cond = threading.Condition(self._gen0_lock)

        self._ready_conns: List[ClientConnection] = []
        self._ready_lock = threading.Lock()
        self._ready_cond = threading.Condition(self._ready_lock)

        self._checkpoint = controller.training_params.samples_per_window()  # gen-0 checkpoint
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_cond = threading.Condition(self._checkpoint_lock)

        # The length of the master_list can be computed on-demand by reading the database. To
        # avoid doing this repeatedly, we grab the value once at start-up, store it as a member, and
        # then update it manually whenever we add new games to the database.
        self._master_list_length: Optional[int] = None  # protected by _pending_game_data_lock
        self._pending_game_data = []
        self._n_pending_rows = 0
        self._pending_game_data_lock = threading.Lock()

    def setup(self):
        self._master_list_length = self._fetch_num_total_augmented_positions()

    def get_num_positions(self):
        return self._master_list_length

    def run_gen0_if_necessary(self):
        additional_gen0_rows_needed = self._num_additional_gen0_positions_needed()
        if additional_gen0_rows_needed == 0:
            self._gen0_complete = True
            return

        with self._ready_lock:
            self._ready_cond.wait_for(lambda: self._ready_conns)
            conn = self._ready_conns[0]

        self._launch_gen0_self_play(conn, additional_gen0_rows_needed)

        with self._gen0_cond:
            self._gen0_cond.wait_for(lambda: self._gen0_complete)

    def run_until_checkpoint(self):
        self._checkpoint = self._controller.get_next_checkpoint()

        if self._checkpoint <= self._master_list_length:
            return

        logger.info('Waiting for more training data... (current=%s, needed=%s)',
                    self._master_list_length, self._checkpoint)

        with self._ready_lock:
            for conn in self._ready_conns:
                if not conn.aux.get('launched', False):
                    self._launch_self_play(conn)

        logger.debug('Unhijacking all self-play tables...')
        self._controller.unhijack_all_self_play_tables()
        with self._checkpoint_lock:
            self._checkpoint_cond.wait_for(lambda: self._master_list_length >= self._checkpoint)
        logger.debug('Hijacking all self-play tables...')
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

    def _fetch_num_total_augmented_positions(self) -> int:
        with self._controller.self_play_db_conn_pool.db_lock:
            # Return cumulative_augmented_positions for the last row of games:
            cursor = self._controller.self_play_db_conn_pool.get_cursor()
            cursor.execute("""SELECT cumulative_augmented_positions FROM games
                           ORDER BY id DESC LIMIT 1""")
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

        if msg_type != 'game':
            # logging every game is too spammy
            logger.debug('self-play-worker received json message: %s', msg)

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'weights-request':
            self._handle_weights_request(conn)
        elif msg_type == 'metrics':
            self._handle_metrics(msg, conn)
        elif msg_type == 'game':
            self._handle_game(msg, conn)
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
                'SELECT gen, cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return total_needed
        gen, n_augmented_positions = row
        if gen > 0:
            return 0

        assert gen == 0, gen
        return max(0, total_needed - n_augmented_positions)

    def _insert_metrics(self, client_id, gen, timestamp, metrics, cursor):
        cache_hits = metrics['cache_hits']
        cache_misses = metrics['cache_misses']
        positions_evaluated = metrics['positions_evaluated']
        batches_evaluated = metrics['batches_evaluated']
        full_batches_evaluated = metrics['full_batches_evaluated']

        cursor.execute(
            'INSERT INTO metrics (client_id, gen, report_timestamp, '
            'cache_hits, cache_misses, positions_evaluated, batches_evaluated, '
            'full_batches_evaluated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                client_id,
                gen,
                timestamp,
                cache_hits,
                cache_misses,
                positions_evaluated,
                batches_evaluated,
                full_batches_evaluated,
            )
        )

        cursor.execute("""UPDATE self_play_metadata
            SET positions_evaluated = positions_evaluated + ?,
                batches_evaluated = batches_evaluated + ?
            WHERE gen = ?""", (positions_evaluated, batches_evaluated, gen))

    def _handle_metrics(self, msg, conn: ClientConnection):
        client_id = conn.client_id
        gen = msg['gen']
        timestamp = msg['timestamp']
        metrics = msg['metrics']

        with self._controller.self_play_db_conn_pool.db_lock:
            db_conn = self._controller.self_play_db_conn_pool.get_connection()
            cursor = db_conn.cursor()
            self._flush_pending_games(conn, cursor)
            self._insert_metrics(client_id, gen, timestamp, metrics, cursor)
            cursor.close()
            db_conn.commit()

        if self._master_list_length >= self._checkpoint:
            with self._checkpoint_lock:
                self._checkpoint_cond.notify_all()

    def _handle_game(self, msg, conn: ClientConnection):
        client_id = conn.client_id
        gen = msg['gen']
        start_timestamp = msg['start_timestamp']
        end_timestamp = msg['end_timestamp']
        rows = msg['rows']
        flush = msg['flush']
        done = msg['done']
        no_file = msg.get('no-file', False)

        with self._pending_game_data_lock:
            use_data = self._master_list_length + self._n_pending_rows < self._checkpoint
            if use_data:
                self._pending_game_data.append((client_id, gen, start_timestamp, end_timestamp,
                                                rows))
                self._n_pending_rows += rows

        if use_data:
            organizer = self._controller.organizer
            # json msg is immediately followed by the game file
            game_dir = organizer.get_self_play_data_dir(gen, client_id)
            os.makedirs(game_dir, exist_ok=True)
            game_filename = os.path.join(game_dir, f'{end_timestamp}.log')
        else:
            # TODO: This is not the best way to prevent overflowing the checkpoint.
            # This makes it so that the c++ blindly keeps generating games, but the python side
            # just drops them on the floor if they exceed the budget. It would be smarter to have
            # the c++ side be aware of the checkpoint.
            game_filename = None

        if not no_file:
            conn.socket.recv_file(game_filename)

        if flush:
            metrics = msg.get('metrics', None)

            with self._controller.self_play_db_conn_pool.db_lock:
                db_conn = self._controller.self_play_db_conn_pool.get_connection()
                cursor = db_conn.cursor()
                self._flush_pending_games(conn, cursor)
                if metrics:
                    self._insert_metrics(client_id, gen, end_timestamp, metrics, cursor)
                cursor.close()
                db_conn.commit()

            if self._master_list_length >= self._checkpoint:
                with self._checkpoint_lock:
                    self._checkpoint_cond.notify_all()

        if done:
            logger.info('Client %s has finished self-play', client_id)
            conn.socket.send_json({'type': 'quit'})

    def _flush_pending_games_helper(self, cursor: sqlite3.Cursor, gen: Generation,
                                    n_games: int, n_augmented_positions: int, runtime: int):
        cursor.execute(
            'INSERT OR IGNORE INTO self_play_metadata (gen) VALUES (?)', (gen,))

        cursor.execute("""UPDATE self_play_metadata
                SET games = games + ?,
                    augmented_positions = augmented_positions + ?,
                    runtime = runtime + ?
                WHERE gen = ?""", (n_games, n_augmented_positions, runtime, gen))

    def _flush_pending_games(self, conn: ClientConnection, cursor: sqlite3.Cursor):
        """
        Flushes pending games to the database, and returns the number of augmented positions.

        Commits to the database unless cursor is provided, in which case the commit is left to the
        caller.
        """
        with self._pending_game_data_lock:
            self._master_list_length += self._n_pending_rows
            pending_game_data_copy = [game_data for game_data in self._pending_game_data]
            self._pending_game_data = []
            self._n_pending_rows = 0

        if not pending_game_data_copy:
            return

        n_games_per_gen: Dict[Generation, int] = defaultdict(int)
        n_augmented_positions_per_gen: Dict[Generation, int] = defaultdict(int)
        for game_data in pending_game_data_copy:
            gen = game_data[1]
            n_rows = game_data[4]
            n_games_per_gen[gen] += 1
            n_augmented_positions_per_gen[gen] += n_rows

        cond = conn.aux['ack_cond']
        with cond:
            conn_gen = conn.aux.get('gen', None)
            start_ts = conn.aux.get('start_ts', None)
            total_runtime = conn.aux.get('total_runtime', 0)
            if start_ts is not None:
                now = time.time_ns()
                elapsed = now - start_ts
                total_runtime += elapsed
                conn.aux['start_ts'] = now

            if 'total_runtime' in conn.aux:
                del conn.aux['total_runtime']

        for gen, n_games in n_games_per_gen.items():
            gen_runtime = total_runtime if gen == conn_gen else 0
            n_augmented_positions = n_augmented_positions_per_gen[gen]
            self._flush_pending_games_helper(cursor, gen, n_games, n_augmented_positions,
                                             gen_runtime)

        cursor.executemany('INSERT INTO games (client_id, gen, start_timestamp, end_timestamp, augmented_positions) VALUES (?, ?, ?, ?, ?)',
                           pending_game_data_copy)

    def _handle_weights_request(self, conn: ClientConnection):
        thread = threading.Thread(target=self._manage_worker, args=(conn,),
                                  daemon=True, name=f'manage-self-play-worker')
        thread.start()
