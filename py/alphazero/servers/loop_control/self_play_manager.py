from .gpu_contention_table import GpuContentionTable
from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ClientId, Generation
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketSendException

from collections import defaultdict
import logging
import os
import sqlite3
import threading
from typing import Dict, Optional, Set


logger = get_logger()


class SelfPlayManager:
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller

        self._gen0_owner: Optional[ClientId] = None
        self._gen0_complete = False
        self._gen0_lock = threading.Lock()
        self._gen0_cond = threading.Condition(self._gen0_lock)

        self._gen1_model_trained = False
        self._gen1_lock = threading.Lock()
        self._gen1_cond = threading.Condition(self._gen1_lock)

        self._pending_game_data = []

    def wait_for_gen0_completion(self):
        self._set_gen0_completion(self._num_additional_gen0_positions_needed() == 0, log=False)
        with self._gen0_cond:
            self._gen0_cond.wait_for(lambda: self._gen0_complete)

    def add_server(self, conn: ClientConnection):
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self._controller.game_spec.name,
        }
        conn.socket.send_json(reply)
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

    def notify_of_new_model(self):
        gen = self._controller.latest_gen()
        if gen > 1:
            return

        with self._gen1_lock:
            self._gen1_model_trained = True
            self._gen1_cond.notify_all()

    def _handle_server_disconnect(self, conn: ClientConnection):
        with self._gen0_lock:
            if conn.client_id == self._gen0_owner:
                self._gen0_owner = None
                self._set_gen0_completion(self._num_additional_gen0_positions_needed() == 0)
                self._gen0_cond.notify_all()

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
        if msg_type != 'log' and logger.isEnabledFor(logging.DEBUG):
            # no need to double-log log-messages
            logger.debug(f'self-play-server received json message: {msg}')

        if msg_type == 'log':
            self._controller.handle_log_msg(msg, conn)
        elif msg_type == 'ready':
            self._handle_ready(conn)
        elif msg_type == 'worker-exit':
            self._controller.handle_worker_exit(msg, conn)
        elif msg_type == 'gen0-complete':
            self._handle_gen0_complete(conn)
        else:
            logger.warn(f'self-play-server: unknown message type: {msg}')
        return False

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']

        if msg_type != 'game' and logger.isEnabledFor(logging.DEBUG):
            # logging every game is too spammy
            logger.debug(f'self-play-worker received json message: {msg}')

        if msg_type == 'log':
            self._controller.handle_log_msg(msg, conn)
        elif msg_type == 'pause-ack':
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
            logger.warn(f'self-play-worker: unknown message type: {msg}')
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

    def _launch_gen0_if_necessary(self, conn: ClientConnection):
        """
        Launches gen0 if necessary. Returns True if gen0 was launched.

        If gen0 is currently being run by a different client, this blocks until that run is
        complete.

        Otherwise, if gen0 has not yet been successfully completed, this launches it and returns
        True. In this case, the call will acquire self._gen0_lock without releasing it.

        Finally, if gen0 has already successfully completed, this returns False.
        """
        with self._gen0_lock:
            self._gen0_cond.wait_for(lambda: self._gen0_owner is None)
            if self._gen0_complete:
                return False

            additional_gen0_rows_needed = self._num_additional_gen0_positions_needed()
            if additional_gen0_rows_needed == 0:
                self._set_gen0_completion(True)
                self._gen0_cond.notify_all()
                return False
            self._gen0_owner = conn.client_id

        self._launch_gen0_self_play(conn, additional_gen0_rows_needed)
        return True

    def _launch_gen0_self_play(self, conn: ClientConnection, num_rows: int):
        logger.info(f'Requesting {conn} to perform gen-0 self-play...')

        data = {
            'type': 'start-gen0',
            'max_rows': num_rows,
        }

        conn.socket.send_json(data)

    def _launch_self_play(self, conn: ClientConnection):
        organizer = self._controller.organizer
        with self._gen1_lock:
            if not self._gen1_model_trained:
                self._gen1_model_trained = organizer.get_latest_model_generation() >= 1
                self._gen1_cond.wait_for(lambda: self._gen1_model_trained)

        gen = organizer.get_latest_model_generation()
        model_filename = organizer.get_model_filename(gen)
        assert gen > 0, gen
        assert os.path.isfile(model_filename), model_filename

        data = {
            'type': 'start',
        }

        logger.info(f'Requesting {conn} to launch self-play...')
        conn.socket.send_json(data)

    def _handle_ready(self, conn: ClientConnection):
        if self._launch_gen0_if_necessary(conn):
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
            logger.warn(f'Error sending to {conn} - worker likely disconnected')
        except:
            logger.error(f'Unexpected error managing {conn}', exc_info=True)
            self._controller.request_shutdown(1)

    def _pause(self, conn: ClientConnection):
        logger.debug(f'Pausing {conn}...')
        data = {
            'type': 'pause',
        }
        conn.aux['pending_pause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_pause_ack' not in conn.aux)

        logger.debug(f'Pause of {conn} complete!')

    def _unpause(self, conn: ClientConnection):
        logger.debug(f'Unpausing {conn}...')
        data = {
            'type': 'unpause',
        }
        conn.aux['pending_unpause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_unpause_ack' not in conn.aux)

        logger.debug(f'Unpause of {conn} complete!')

    def _handle_pause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_pause_ack', None)
            cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

    def _refresh_weights_if_needed(self, conn: ClientConnection):
        gen = self._controller.latest_gen()
        if conn.aux.get('gen', None) != gen:
            self._controller.broadcast_weights([conn], gen)
            conn.aux['gen'] = gen

    def _handle_gen0_complete(self, conn: ClientConnection):
        with self._gen0_lock:
            assert conn.client_id == self._gen0_owner
            assert self._num_additional_gen0_positions_needed() == 0
            self._gen0_owner = None
            self._set_gen0_completion(True)
            self._gen0_cond.notify_all()

        self._launch_self_play(conn)

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
            conn = self._controller.self_play_db_conn_pool.get_connection()
            cursor = conn.cursor()
            n_augmented_positions = self._flush_pending_games(client_id, cursor)
            self._insert_metrics(client_id, gen, timestamp, metrics, cursor)
            cursor.close()
            conn.commit()
        self._controller.handle_new_self_play_positions(n_augmented_positions)

    def _handle_game(self, msg, conn: ClientConnection):
        client_id = conn.client_id
        gen = msg['gen']
        start_timestamp = msg['start_timestamp']
        end_timestamp = msg['end_timestamp']
        rows = msg['rows']
        flush = msg['flush']

        organizer = self._controller.organizer
        # json msg is immediately followed by the game file
        game_dir = os.path.join(organizer.self_play_data_dir, f'client-{client_id}',
                                f'gen-{gen}')
        os.makedirs(game_dir, exist_ok=True)
        game_filename = os.path.join(game_dir, f'{end_timestamp}.pt')
        conn.socket.recv_file(game_filename)

        self._pending_game_data.append((client_id, gen, start_timestamp, end_timestamp, rows))

        if flush:
            metrics = msg.get('metrics', None)

            with self._controller.self_play_db_conn_pool.db_lock:
                conn = self._controller.self_play_db_conn_pool.get_connection()
                cursor = conn.cursor()
                n_augmented_positions = self._flush_pending_games(client_id, cursor)
                if metrics:
                    self._insert_metrics(client_id, gen, end_timestamp, metrics, cursor)
                cursor.close()
                conn.commit()
            self._controller.handle_new_self_play_positions(n_augmented_positions)

    def _flush_pending_games_helper(self, cursor: sqlite3.Cursor, gen: Generation,
                                    game_data: Dict[Generation, list]):
        cursor.execute(
            'INSERT OR IGNORE INTO self_play_metadata (gen) VALUES (?)', (gen,))

        game_subdict = defaultdict(list)  # keyed by client_id
        n_games = 0
        n_augmented_positions = 0
        for client_id, start_timestamp, end_timestamp, n_rows in game_data:
            game_subdict[client_id].append((start_timestamp, end_timestamp))
            n_games += 1
            n_augmented_positions += n_rows

        cursor.execute("""UPDATE self_play_metadata
                SET games = games + ?,
                    augmented_positions = augmented_positions + ?
                WHERE gen = ?""", (n_games, n_augmented_positions, gen))

        for client_id, game_subdata in game_subdict.items():
            start_timestamp = min([x[0] for x in game_subdata])
            end_timestamp = max([x[1] for x in game_subdata])

            cursor.execute('INSERT OR IGNORE INTO timestamps (gen, client_id, start_timestamp, end_timestamp) VALUES (?, ?, ?, ?)',
                      (gen, client_id, start_timestamp, end_timestamp))
            cursor.execute("""UPDATE timestamps
                    SET start_timestamp = MIN(start_timestamp, ?),
                        end_timestamp = MAX(end_timestamp, ?)
                    WHERE gen = ? AND client_id = ?""",
                      (start_timestamp, end_timestamp, gen, client_id))

    def _flush_pending_games(self, client_id: ClientId, cursor: sqlite3.Cursor):
        """
        Flushes pending games to the database, and returns the number of augmented positions.

        Commits to the database unless cursor is provided, in which case the commit is left to the
        caller.
        """
        if not self._pending_game_data:
            return 0

        n_augmented_positions = 0
        game_dict = defaultdict(list)  # keyed by gen
        for client_id, gen, start_timestamp, end_timestamp, n_rows in self._pending_game_data:
            n_augmented_positions += n_rows
            game_dict[gen].append(
                (client_id, start_timestamp, end_timestamp, n_rows))

        for gen, game_data in game_dict.items():
            self._flush_pending_games_helper(cursor, gen, game_data)

        cursor.executemany('INSERT INTO games (client_id, gen, start_timestamp, end_timestamp, augmented_positions) VALUES (?, ?, ?, ?, ?)',
                           self._pending_game_data)
        self._pending_game_data = []
        return n_augmented_positions

    def _handle_weights_request(self, conn: ClientConnection):
        thread = threading.Thread(target=self._manage_worker, args=(conn,),
                                  daemon=True, name=f'manage-self-play-worker')
        thread.start()
