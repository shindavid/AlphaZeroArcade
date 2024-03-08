from alphazero.logic.aux_subcontroller import AuxSubcontroller, NewModelSubscriber
from alphazero.logic.custom_types import ClientData, ClientId, Generation
from alphazero.logic.loop_control_data import LoopControlData
from alphazero.logic.training_subcontroller import TrainingSubcontroller
from util.logging_util import get_logger
from util.socket_util import send_json, JsonDict

from collections import defaultdict
import logging
import os
import sqlite3
import threading
from typing import Dict, Optional


logger = get_logger()


class SelfPlaySubcontroller(NewModelSubscriber):
    """
    Used by the LoopController to manage self-play. The actual self-play is performed in external
    servers; this subcontroller acts as a sort of remote-manager of those servers.
    """

    def __init__(self, training_controller: TrainingSubcontroller):
        self.training_controller = training_controller
        self.aux_controller.subscribe_to_new_model_announcements(self)

        self._gen0_owner: Optional[ClientId] = None
        self._gen0_complete = False
        self._gen0_lock = threading.Lock()
        self._gen0_cv = threading.Condition(self._gen0_lock)

        self._gen1_model_trained = False
        self._gen1_lock = threading.Lock()
        self._gen1_cv = threading.Condition(self._gen1_lock)

        self._pending_game_data = []

    def handle_new_model(self, generation: Generation):
        if generation > 1:
            return

        with self._gen1_lock:
            self._gen1_model_trained = True
            self._gen1_cv.notify_all()

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
        self.aux_controller.launch_recv_loop(
            self.manager_msg_handler, client_data, 'self-play-manager',
            disconnect_handler=self.handle_manager_disconnect)

    def add_self_play_worker(self, client_data: ClientData):
        reply = {
            'type': 'handshake_ack',
            'client_id': client_data.client_id,
        }
        send_json(client_data.sock, reply)
        self.aux_controller.launch_recv_loop(
            self.worker_msg_handler, client_data, 'self-play-worker')

    def handle_manager_disconnect(self, client_data: ClientData):
        with self._gen0_lock:
            if client_data.client_id == self._gen0_owner:
                self._gen0_owner = None
                self._set_gen0_completion(self.num_additional_gen0_positions_needed() == 0)
                self._gen0_cv.notify_all()

    def manager_msg_handler(self, client_data: ClientData, msg: JsonDict) -> bool:
        msg_type = msg['type']
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'self-play-manager received json message: {msg}')

        if msg_type == 'asset_request':
            self.aux_controller.send_asset(msg['asset'], client_data)
        elif msg_type == 'ready':
            self.handle_ready(client_data)
        elif msg_type == 'gen0-complete':
            self.handle_gen0_complete(client_data)
        return False

    def worker_msg_handler(self, client_data: ClientData, msg: JsonDict) -> bool:
        msg_type = msg['type']

        if msg_type != 'game' and logger.isEnabledFor(logging.DEBUG):
            # logging every game is too spammy
            logger.debug(f'self-play-worker received json message: {msg}')

        if msg_type == 'pause_ack':
            self.aux_controller.handle_pause_ack(client_data)
        elif msg_type == 'metrics':
            self.handle_metrics(msg, client_data)
        elif msg_type == 'game':
            self.handle_game(msg, client_data)
        elif msg_type == 'done':
            return True
        return False

    def _set_gen0_completion(self, complete: bool):
        """
        Assumes self._gen0_lock is already acquired.
        """
        if self._gen0_complete:
            return

        self._gen0_complete = complete
        if complete:
            logger.info('Gen-0 self-play complete!')

    def wait_for_gen0_completion(self):
        with self._gen0_cv:
            self._gen0_cv.wait_for(lambda: self._gen0_complete)

    def launch_gen0_if_necessary(self, client_data: ClientData):
        """
        Launches gen0 if necessary. Returns True if gen0 was launched.

        If gen0 is currently being run by a different client, this blocks until that run is
        complete.

        Otherwise, if gen0 has not yet been successfully completed, this launches it and returns
        True. In this case, the call will acquire self._gen0_lock without releasing it.

        Finally, if gen0 has already successfully completed, this returns False.
        """
        with self._gen0_lock:
            self._gen0_cv.wait_for(lambda: self._gen0_owner is None)
            if self._gen0_complete:
                return False

            additional_gen0_rows_needed = self.num_additional_gen0_positions_needed()
            if additional_gen0_rows_needed == 0:
                self._set_gen0_completion(True)
                self._gen0_cv.notify_all()
                return False
            self._gen0_owner = client_data.client_id

        self.launch_gen0_self_play(client_data, additional_gen0_rows_needed)
        return True

    def launch_gen0_self_play(self, client_data: ClientData, num_rows: int):
        logger.info(f'Requesting {client_data} to perform gen-0 self-play...')

        data = {
            'type': 'start-gen0',
            'games_base_dir': self.data.organizer.self_play_data_dir,
            'max_rows': num_rows,
        }

        send_json(client_data.sock, data)

    def launch_self_play(self, client_data: ClientData):
        with self._gen1_lock:
            if not self._gen1_model_trained:
                self._gen1_model_trained = self.data.organizer.get_latest_generation() >= 1
                self._gen1_cv.wait_for(lambda: self._gen1_model_trained)

        gen = self.data.organizer.get_latest_model_generation()
        model_filename = self.data.organizer.get_model_filename(gen)
        assert gen > 0, gen
        assert os.path.isfile(model_filename), model_filename

        data = {
            'type': 'start',
            'gen': gen,
            'games_base_dir': self.data.organizer.self_play_data_dir,
            'model': model_filename,
        }

        logger.info(f'Requesting {client_data} to launch self-play...')
        send_json(client_data.sock, data)

    def handle_ready(self, client_data: ClientData):
        if self.launch_gen0_if_necessary(client_data):
            return

        self.launch_self_play(client_data)

    def handle_gen0_complete(self, client_data: ClientData):
        with self._gen0_lock:
            assert client_data.client_id == self._gen0_owner
            assert self.num_additional_gen0_positions_needed() == 0
            self._gen0_owner = None
            self._set_gen0_completion(True)
            self._gen0_cv.notify_all()

        self.launch_self_play(client_data)

    def num_additional_gen0_positions_needed(self) -> int:
        total_needed = self.data.training_params.samples_per_window()
        with self.data.self_play_db_conn_pool.db_lock:
            cursor = self.data.self_play_db_conn_pool.get_cursor()
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
