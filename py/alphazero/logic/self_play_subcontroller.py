from alphazero.logic.aux_subcontroller import AuxSubcontroller
from alphazero.logic.custom_types import ClientData
from alphazero.logic.custom_types import ClientType, Generation
from alphazero.logic.loop_control_data import LoopControlData
from alphazero.logic.training_subcontroller import TrainingSubcontroller
from util.logging_util import get_logger
from util.socket_util import recv_json, send_json

from collections import defaultdict
import sqlite3
import threading
from typing import Dict


logger = get_logger()


class SelfPlaySubcontroller:
    """
    Used by the LoopController to manage self-play. The actual self-play is performed in external
    servers; this subcontroller acts as a sort of remote-manager of those servers.
    """

    def __init__(self, training_controller: TrainingSubcontroller):
        self.training_controller = training_controller

        self._done_event = threading.Event()
        self._connected = False
        self._connected_event = threading.Event()
        self._connected_lock = threading.Lock()

        self._pending_game_data = []

    @property
    def aux_controller(self) -> AuxSubcontroller:
        return self.training_controller.aux_controller

    @property
    def data(self) -> LoopControlData:
        return self.training_controller.data

    def add_self_play_manager(self, client_data: ClientData):
        reply = {
            'type': 'handshake_ack',
            'client_id': client_data.client_id,
        }
        self.aux_controller.add_asset_metadata_to_reply(reply)
        send_json(client_data.sock, reply)
        threading.Thread(target=self.manager_recv_loop, name='self-play-manager-recv-loop',
                            args=(client_data,), daemon=True).start()

    def add_self_play_worker(self, client_data: ClientData):
        reply = {
            'type': 'handshake_ack',
            'client_id': client_data.client_id,
        }
        send_json(client_data.sock, reply)
        threading.Thread(target=self.worker_recv_loop, name='self-play-worker-recv-loop',
                            args=(client_data,), daemon=True).start()

    def manager_recv_loop(self, client_data: ClientData):
        try:
            while True:
                try:
                    msg = recv_json(client_data.sock)
                except OSError:
                    self.aux_controller.handle_disconnect(client_data)
                    return

                msg_type = msg['type']
                if msg_type == 'asset_request':
                    self.aux_controller.send_asset(msg['asset'], client_data)
                elif msg_type == 'ready':
                    self.handle_ready()
        except:
            logger.error(
                f'Unexpected error in SelfPlaySubcontroller.self_play_manager_loop({client_data}):',
                exc_info=True)
            self.data.signal_error()

    def worker_recv_loop(self, client_data: ClientData):
        try:
            while True:
                try:
                    msg = recv_json(client_data.sock)
                except OSError:
                    self.aux_controller.handle_disconnect(client_data)
                    return

                msg_type = msg['type']
                if msg_type == 'pause_ack':
                    self.aux_controller.handle_pause_ack(client_data)
                elif msg_type == 'metrics':
                    self.handle_metrics(msg, client_data)
                elif msg_type == 'game':
                    self.handle_game(msg, client_data)
                elif msg_type == 'done':
                    self.aux_controller.handle_disconnect(client_data)
                    self._done_event.set()
                    break
        except:
            logger.error(
                f'Unexpected error in SelfPlaySubcontroller.self_play_worker_loop({client_data}):',
                exc_info=True)
            self.data.signal_error()

    def handle_ready(self):
        with self._connected_lock:
            if not self._connected:
                self._connected = True
                self._connected_event.set()

    def launch(self):
        threading.Thread(target=self.launch_helper, name='self-play-launch').start()

    def launch_helper(self):
        gen = self.data.organizer.get_latest_model_generation()
        model_filename = self.data.organizer.get_model_filename(gen)

        self.wait_for_connection()

        data = {
            'type': 'start',
            'gen': gen,
            'games_base_dir': self.data.organizer.self_play_data_dir,
            'model': model_filename,
        }

        for client_data in self.data.get_client_data_list(ClientType.SELF_PLAY_MANAGER):
            logger.info(f'Requesting {client_data} to launch self-play...')
            send_json(client_data.sock, data)

    def is_gen0_complete(self) -> bool:
        """
        Returns True if the first generation has been completed, False otherwise.
        """
        with self.data.self_play_db_conn_pool.db_lock:
            cursor = self.data.self_play_db_conn_pool.get_cursor()
            cursor.execute(
                'SELECT gen, cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return False
        gen, n_augmented_positions = row
        if gen > 0:
            return True

        assert gen == 0, gen
        return n_augmented_positions >= self.data.training_params.samples_per_window()

    def wait_for_connection(self):
        if self._connected_event.is_set():
            return

        logger.info('Waiting for self play client connection...')
        self._connected_event.wait()
        logger.info('Self play client connected!')

    def run_gen0_if_necessary(self):
        if self.is_gen0_complete():
            return

        self.wait_for_connection()

        client_data = self.data.get_single_client_data(ClientType.SELF_PLAY_MANAGER)
        logger.info(f'Requesting {client_data} to perform gen-0 self-play...')
        max_rows = self.data.training_params.samples_per_window()

        data = {
            'type': 'start-gen0',
            'games_base_dir': self.data.organizer.self_play_data_dir,
            'max_rows': max_rows,
        }

        send_json(client_data.sock, data)
        self._done_event.wait()
        self._done_event.clear()
        logger.info(f'Gen-0 self-play complete!')

    def insert_metrics(self, client_id, gen, timestamp, metrics, cursor):
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

    def handle_metrics(self, msg, client_data: ClientData):
        client_id = client_data.client_id
        gen = msg['gen']
        timestamp = msg['timestamp']
        metrics = msg['metrics']

        with self.data.self_play_db_conn_pool.db_lock:
            conn = self.data.self_play_db_conn_pool.get_connection()
            cursor = conn.cursor()
            n_augmented_positions = self.flush_pending_games(client_id, cursor)
            self.insert_metrics(client_id, gen, timestamp, metrics, cursor)
            cursor.close()
            conn.commit()
        self.training_controller.increment_master_list_length(n_augmented_positions)

    def handle_game(self, msg, client_data: ClientData):
        client_id = client_data.client_id
        gen = msg['gen']
        start_timestamp = msg['start_timestamp']
        end_timestamp = msg['end_timestamp']
        rows = msg['rows']
        flush = msg['flush']

        self._pending_game_data.append(
            (client_id, gen, start_timestamp, end_timestamp, rows))

        if flush:
            metrics = msg.get('metrics', None)

            with self.data.self_play_db_conn_pool.db_lock:
                conn = self.data.self_play_db_conn_pool.get_connection()
                cursor = conn.cursor()
                n_augmented_positions = self.flush_pending_games(client_id, cursor)
                if metrics:
                    self.insert_metrics(client_id, gen, end_timestamp, metrics, cursor)
                cursor.close()
                conn.commit()
            self.training_controller.increment_master_list_length(n_augmented_positions)

    def _flush_pending_games_helper(self, c: sqlite3.Cursor, gen: Generation,
                                    game_data: Dict[Generation, list]):
        c.execute(
            'INSERT OR IGNORE INTO self_play_metadata (gen) VALUES (?)', (gen,))

        game_subdict = defaultdict(list)  # keyed by client_id
        n_games = 0
        n_augmented_positions = 0
        for client_id, start_timestamp, end_timestamp, n_rows in game_data:
            game_subdict[client_id].append((start_timestamp, end_timestamp))
            n_games += 1
            n_augmented_positions += n_rows

        c.execute("""UPDATE self_play_metadata
                SET games = games + ?,
                    augmented_positions = augmented_positions + ?
                WHERE gen = ?""", (n_games, n_augmented_positions, gen))

        for client_id, game_subdata in game_subdict.items():
            start_timestamp = min([x[0] for x in game_subdata])
            end_timestamp = max([x[1] for x in game_subdata])

            c.execute('INSERT OR IGNORE INTO timestamps (gen, client_id, start_timestamp, end_timestamp) VALUES (?, ?, ?, ?)',
                      (gen, client_id, start_timestamp, end_timestamp))
            c.execute("""UPDATE timestamps
                    SET start_timestamp = MIN(start_timestamp, ?),
                        end_timestamp = MAX(end_timestamp, ?)
                    WHERE gen = ? AND client_id = ?""",
                      (start_timestamp, end_timestamp, gen, client_id))

    def flush_pending_games(self, client_id, cursor):
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
