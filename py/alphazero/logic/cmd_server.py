from alphazero.logic.sample_window_logic import SamplingParams, Window, construct_window, \
    get_required_dataset_size
from alphazero.logic import constants
from alphazero.logic.common_params import CommonParams
from alphazero.logic.custom_types import ChildThreadError, Generation
from alphazero.logic.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger
from util.socket_util import JsonDict, recv_json, send_json

import argparse
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import os
import signal
import socket
import sqlite3
import threading
from typing import Any, Dict, List, Optional
import sys


ClientId = int
ThreadId = int


logger = get_logger()


class ClientType(Enum):
    SELF_PLAY_WRAPPER = 'self-play-wrapper'
    SELF_PLAY = 'self-play'
    TRAINING = 'training'


@dataclass
class ClientData:
    client_type: ClientType
    client_id: ClientId
    sock: socket.socket
    start_timestamp: int
    cuda_device: str  # empty str if no cuda device

    @property
    def ip_address(self):
        return self.sock.getsockname()[0]

    @property
    def port(self):
        return self.sock.getsockname()[1]

    def shares_gpu_with(self, other: 'ClientData') -> bool:
        if self.cuda_device == '' or other.cuda_device == '':
            return False
        if self.ip_address != other.ip_address:
            return False
        return self.cuda_device == other.cuda_device

    def __str__(self):
        tokens = [str(self.client_type), str(self.client_id),
                  f'{self.ip_address}:{self.port}', self.cuda_device]
        tokens = [t for t in tokens if t]
        return f'ClientData({", ".join(tokens)})'


@dataclass
class CmdServerParams:
    port: int = constants.DEFAULT_CMD_SERVER_PORT

    @staticmethod
    def create(args) -> 'CmdServerParams':
        return CmdServerParams(
            port=args.port,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CmdServer options')
        defaults = CmdServerParams()

        group.add_argument('--port', type=int, default=defaults.port,
                           help='port (default: %(default)s)')


class CmdServer:
    """
    The cmd server coordinates activity between the self-play server and the training server.

    The eventual goal is for the cmd server to exclusively access the filesystem, while the
    self-play and training servers communicate with the cmd-server over TCP. This will allow us
    to flexibly distribute the self-play and training servers across multiple machines. Currently,
    game data and models/checkpoints are written to the filesystem by the self-play and training
    servers, so this goal has not yet been achieved.
    """

    def __init__(self, params: CmdServerParams, common_params: CommonParams,
                 sampling_params: SamplingParams):
        self.organizer = DirectoryOrganizer(common_params)
        self.organizer.makedirs()
        self.host = 'localhost'
        self.port = params.port
        self.sampling_params = sampling_params

        self._pause_ack_events: Dict[ClientId, threading.Event] = {}
        self._client_data_list: List[ClientData] = []
        self._client_data_lock = threading.Lock()
        self._db_conn_dict: Dict[ThreadId, sqlite3.Connection] = {}

        self._pending_game_data = []

        self._server_socket = None
        self._last_sample_window = None
        self._master_list_length = None
        self._master_list_length_for_next_train_loop = None

        self._shutdown_code = None
        self._child_thread_error_flag = threading.Event()

        self._train_ready_event = threading.Event()
        self._train_ready_lock = threading.Lock()

        self._self_play_done_event = threading.Event()
        self._train_step_done_event = threading.Event()

        self._self_play_client_connected = False
        self._training_client_connected = False
        self._self_play_client_connected_event = threading.Event()
        self._training_client_connected_event = threading.Event()

    def register_signal_handler(self):
        def signal_handler(sig, frame):
            logger.info('Detected Ctrl-C.')
            self.shutdown(0)

        signal.signal(signal.SIGINT, signal_handler)

    def _run_setup(self):
        logger.info('Performing CmdServer setup...')
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.setblocking(True)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen()

        self._last_sample_window: Window = self.load_last_sample_window()

        # The length of the master_list can be computed on-demand by reading the database. To
        # avoid doing this repeatedly, we grab the value once at start-up, store it as a member, and
        # then update it manually whenever we add new games to the database.
        self._master_list_length = self.compute_master_list_length()

        # This is the length that the master_list needs to be before we can start a new train loop.
        # Initialized lazily.
        self._master_list_length_for_next_train_loop = 0

        logger.info(f'Listening for CmdServer clients on port {self.port}...')
        threading.Thread(target=self.accept_clients, daemon=True).start()

    def wait_for_self_play_client_connection(self):
        logger.info('Waiting for self play client connection...')
        while True:
            self._self_play_client_connected_event.wait(timeout=1)
            if self._self_play_client_connected_event.is_set():
                return

            if self._child_thread_error_flag.is_set():
                logger.info('Child thread error caught, exiting wait_for_self_play_client_connection() loop')
                raise ChildThreadError()

    def wait_for_training_client_connection(self):
        logger.info('Waiting for training client connection...')
        while True:
            self._training_client_connected_event.wait(timeout=1)
            if self._training_client_connected_event.is_set():
                return

            if self._child_thread_error_flag.is_set():
                logger.info('Child thread error caught, exiting wait_for_training_client_connection() loop')
                raise ChildThreadError()

    def run(self):
        self._run_setup()

        try:
            self.wait_for_self_play_client_connection()
            self.run_gen0_if_necessary()
            self.wait_for_training_client_connection()
            self.train_gen1_model_if_necessary()
            self.launch_self_play()

            while True:
                self.wait_until_enough_training_data()
                self.train_step()
        except ChildThreadError:
            logger.info('Child thread error caught, exiting run()')
            self.shutdown(1)
        except:
            if self._shutdown_code == 0:
                return

            logger.error('Unexpected error in run():', exc_info=True)
            self.shutdown(1)

    def run_gen0_if_necessary(self):
        if self.is_gen0_complete():
            return

        client_data = self.get_single_client_data(ClientType.SELF_PLAY_WRAPPER)
        logger.info(f'Requesting {client_data} to perform gen-0 self-play...')
        max_rows = self.sampling_params.samples_per_window()

        data = {
            'type': 'start-gen0',
            'games_base_dir': self.organizer.self_play_data_dir,
            'max_rows': max_rows,
            }

        send_json(client_data.sock, data)
        self._self_play_done_event.wait()
        self._self_play_done_event.clear()
        logger.info(f'Gen-0 self-play complete!')

    def train_gen1_model_if_necessary(self):
        gen = 1
        model_filename = self.organizer.get_model_filename(gen)
        if os.path.isfile(model_filename):
            return

        self.train_step()

    def train_step(self):
        gen = self.organizer.get_latest_model_generation() + 1
        logger.info(f'Performing gen-{gen} train step...')

        c = self.my_db_conn.cursor()
        c.execute("""SELECT cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1""")
        row = c.fetchone()
        n = row[0]

        f = self.sampling_params.window_size_function
        n = row[0]
        c = int(n - f(n))

        data = {
            'type': 'train-step',
            'base_dir': self.organizer.base_dir,
            'gen': gen,
            'start': c,
            'end': n,
            'n_minibatches': self.sampling_params.minibatches_per_epoch,
            'minibatch_size': self.sampling_params.minibatch_size,
        }

        client_data = self.get_single_client_data(ClientType.TRAINING)
        send_json(client_data.sock, data)

        self.wait_for_train_step_done()

    def wait_for_train_step_done(self):
        while True:
            self._train_step_done_event.wait(timeout=1)
            if self._train_step_done_event.is_set():
                self._train_step_done_event.clear()
                return

            if self._child_thread_error_flag.is_set():
                logger.info('Child thread error caught, exiting wait_for_train_step_done() loop')
                raise ChildThreadError()

    def handle_train_step_done(self, msg: JsonDict):
        stats = msg['stats']

        gen = stats['gen']
        start_ts = stats['start_ts']
        end_ts = stats['end_ts']
        window_start = stats['window_start']
        window_end = stats['window_end']
        n_samples = stats['n_samples']
        substats = stats['substats']

        window = construct_window(self._last_sample_window, window_start, window_end, n_samples)
        self._last_sample_window = window

        c = self.my_db_conn.cursor()

        c.execute("""INSERT OR REPLACE INTO training (gen, training_start_ts, training_end_ts,
            window_start, window_end, window_sample_rate)
            VALUES (?, ?, ?, ?, ?, ?)""",
                  (gen, start_ts, end_ts, window.start, window.end, window.sample_rate))

        head_data = []
        for head_name, head_stats in substats.items():
            loss = head_stats['loss']
            loss_weight = head_stats['loss_weight']
            accuracy = head_stats['accuracy']
            head_data.append((gen, head_name, loss, loss_weight, accuracy))

        c.executemany("""INSERT OR REPLACE INTO training_heads (gen, head_name, loss, loss_weight, accuracy)
            VALUES (?, ?, ?, ?, ?)""", head_data)

        self.my_db_conn.commit()

        self.reload_weights(gen)
        self._train_step_done_event.set()

    def handle_lock_gpu(self, msg: JsonDict, client_data: ClientData):
        self.pause_shared_gpu_self_play_clients(client_data)

        data = {
            'type': 'lock-gpu-ack',
        }
        send_json(client_data.sock, data)

    def launch_self_play(self):
        client_data = self.get_single_client_data(ClientType.SELF_PLAY_WRAPPER)
        logger.info(f'Requesting {client_data} to perform continuous self-play...')

        gen = self.organizer.get_latest_model_generation()
        model_filename = self.organizer.get_model_filename(gen)

        data = {
            'type': 'start',
            'gen': gen,
            'games_base_dir': self.organizer.self_play_data_dir,
            'model': model_filename,
        }

        send_json(client_data.sock, data)

    def wait_until_enough_training_data(self):
        logger.info('Waiting for more training data...')
        with self._train_ready_lock:
            self._master_list_length_for_next_train_loop = get_required_dataset_size(
                self.sampling_params, self._last_sample_window)
            self._train_ready_event.clear()

        self._train_ready_event.wait()

    @property
    def my_db_conn(self) -> sqlite3.Connection:
        """
        sqlite3 demands a single connection per thread. This property hides this detail under the
        hood.
        """
        thread_id = threading.get_ident()
        conn = self._db_conn_dict.get(thread_id, None)
        if conn is None:
            conn = self._create_db_conn()
            self._db_conn_dict[thread_id] = conn
        return conn

    def close_my_db_conn(self):
        thread_id = threading.get_ident()
        if thread_id in self._db_conn_dict:
            self._db_conn_dict[thread_id].close()
            del self._db_conn_dict[thread_id]

    def is_gen0_complete(self) -> bool:
        """
        Returns True if the first generation has been completed, False otherwise.
        """
        cursor = self.my_db_conn.cursor()
        cursor.execute(
            'SELECT gen, cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        if row is None:
            return False
        gen, n_augmented_positions = row
        if gen > 0:
            return True

        assert gen == 0, gen
        return n_augmented_positions >= self.sampling_params.samples_per_window()

    def load_last_sample_window(self) -> Window:
        cursor = self.my_db_conn.cursor()
        cursor.execute("""SELECT window_start, window_end, window_sample_rate
                          FROM training ORDER BY gen DESC LIMIT 1""")
        row = cursor.fetchone()
        if row is None:
            # kZero-style initialization of sample window
            samples_per_window = self.sampling_params.samples_per_window()
            target_sample_rate = self.sampling_params.target_sample_rate
            return Window(0, samples_per_window, target_sample_rate)
        return Window(*row)

    def compute_master_list_length(self) -> int:
        # Return cumulative_augmented_positions for the last row of games:
        cursor = self.my_db_conn.cursor()
        cursor.execute("""SELECT cumulative_augmented_positions
                       FROM games ORDER BY id DESC LIMIT 1""")
        row = cursor.fetchone()
        if row is None:
            return 0
        return row[0]

    def _create_db_conn(self):
        db_filename = self.organizer.training_db_filename
        if os.path.isfile(db_filename):
            return sqlite3.connect(db_filename)

        conn = sqlite3.connect(db_filename)
        c = conn.cursor()
        c.execute("""CREATE TABLE training (
            gen INTEGER PRIMARY KEY,
            training_start_ts INTEGER,
            training_end_ts INTEGER DEFAULT 0,
            window_start INTEGER,
            window_end INTEGER,
            window_sample_rate FLOAT
            )""")

        c.execute("""CREATE TABLE training_heads (
            gen INTEGER,
            head_name TEXT,
            loss FLOAT,
            loss_weight FLOAT,
            accuracy FLOAT
            )""")

        c.execute("""CREATE TABLE clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            port INTEGER,
            role TEXT,
            start_timestamp INTEGER,
            cuda_device TEXT
            )""")

        c.execute("""CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            gen INTEGER,
            report_timestamp INTEGER,
            cache_hits INTEGER,
            cache_misses INTEGER,
            positions_evaluated INTEGER,
            batches_evaluated INTEGER,
            full_batches_evaluated INTEGER
            )""")

        c.execute("""CREATE TABLE games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            gen INTEGER,
            start_timestamp INTEGER,
            end_timestamp INTEGER,
            augmented_positions INTEGER,
            cumulative_augmented_positions INTEGER
            )""")

        c.execute("""CREATE TABLE self_play_metadata (
            gen INTEGER PRIMARY KEY,
            positions_evaluated INTEGER DEFAULT 0,
            batches_evaluated INTEGER DEFAULT 0,
            games INTEGER DEFAULT 0,
            augmented_positions INTEGER DEFAULT 0
            )""")

        c.execute("""CREATE TABLE timestamps (
            gen INTEGER,
            client_id INTEGER,
            start_timestamp INTEGER DEFAULT 0,
            end_timestamp INTEGER DEFAULT 0,
            PRIMARY KEY (gen, client_id)
            )""")

        c.execute(
            """CREATE INDEX training_heads_idx ON training_heads (gen)""")

        c.execute("""CREATE TRIGGER update_games AFTER INSERT ON games
            BEGIN
                UPDATE games
                SET cumulative_augmented_positions = CASE
                WHEN NEW.id = 1 THEN NEW.augmented_positions
                ELSE (SELECT cumulative_augmented_positions FROM games WHERE id = NEW.id - 1)
                        + NEW.augmented_positions
                END
                WHERE id = NEW.id;
            END""")

        conn.commit()
        return conn

    def remove_client(self, client_id: ClientId):
        with self._client_data_lock:
            self._client_data_list = [c for c in self._client_data_list if c.client_id != client_id]

    def setup_done_check(self):
        with self._client_data_lock:
            types = set()
            for client_data in self._client_data_list:
                types.add(client_data.client_type)

            if not self._training_client_connected:
                if ClientType.TRAINING in types:
                    self._training_client_connected = True
                    self._training_client_connected_event.set()

            if not self._self_play_client_connected:
                if ClientType.SELF_PLAY_WRAPPER in types:
                    self._self_play_client_connected = True
                    self._self_play_client_connected_event.set()

    def accept_clients(self):
        """
        Loop that checks for new clients. For each new client, spawns a thread to handle it.
        """
        db_conn = self.my_db_conn
        while True:
            logger.info('Waiting for new client...')
            client_id = None
            # if True:
            try:
                client_socket, addr = self._server_socket.accept()
                ip_address, port = addr

                msg = recv_json(client_socket)  #, timeout=1)
                assert msg['type'] == 'handshake', f'Expected handshake from client, got {msg}'
                role = msg['role']
                client_type = ClientType(role)

                reply = {'type': 'handshake_ack'}

                if client_type == ClientType.SELF_PLAY_WRAPPER:
                    handler_fn = self.handle_self_play_wrapper_client
                elif client_type == ClientType.SELF_PLAY:
                    handler_fn = self.handle_self_play_client
                elif client_type == ClientType.TRAINING:
                    handler_fn = self.handle_training_client
                    reply['base_dir'] = self.organizer.base_dir
                    reply['db_filename'] = self.organizer.training_db_filename
                else:
                    raise Exception(f'Unknown client type: {client_type}')

                start_timestamp = msg['start_timestamp']
                cuda_device = msg.get('cuda_device', '')

                c = db_conn.cursor()
                c.execute('INSERT INTO clients (ip_address, port, role, start_timestamp, cuda_device) VALUES (?, ?, ?, ?, ?)',
                        (ip_address, port, role, start_timestamp, cuda_device)
                        )
                client_id = c.lastrowid
                db_conn.commit()

                client_data = ClientData(
                    client_type, client_id, client_socket, start_timestamp, cuda_device)

                self._client_data_lock.acquire()
                self._client_data_list.append(client_data)
                self._client_data_lock.release()

                logger.info(f'Accepted client: {client_data}')

                reply['client_id'] = client_id
                send_json(client_socket, reply)
                threading.Thread(target=handler_fn, args=(client_data,), daemon=True).start()

                self.setup_done_check()
            except:
                logger.error('Exception in accept_clients():', exc_info=True)
                self._child_thread_error_flag.set()

    def handle_disconnect(self, client_data: ClientData):
        logger.info(f'Handling disconnect for {client_data}...')
        self.remove_client(client_data.client_id)
        self.close_my_db_conn()
        client_data.sock.close()
        logger.info(f'Disconnect complete!')

    def shutdown(self, return_code):
        logger.info('Shutting down...')
        self._shutdown_code = return_code
        sys.exit(return_code)

    def handle_self_play_wrapper_client(self, client_data: ClientData):
        try:
            while True:
                try:
                    msg = recv_json(client_data.sock)
                except OSError:
                    self.handle_disconnect(client_data)
                    return

                msg_type = msg['type']
                # TODO
        except:
            logger.error(
                f'Unexpected error in handle_self_play_wrapper_client({client_data}):',
                exc_info=True)
            self._child_thread_error_flag.set()

    def handle_self_play_client(self, client_data: ClientData):
        pause_ack_event = threading.Event()
        self._pause_ack_events[client_data.client_id] = pause_ack_event

        try:
            while True:
                try:
                    msg = recv_json(client_data.sock)
                except OSError:
                    self.handle_disconnect(client_data)
                    return

                msg_type = msg['type']
                if msg_type == 'pause_ack':
                    pause_ack_event.set()
                elif msg_type == 'metrics':
                    self.handle_metrics(msg, client_data)
                elif msg_type == 'game':
                    self.handle_game(msg, client_data)
                elif msg_type == 'done':
                    self.handle_disconnect(client_data)
                    self._self_play_done_event.set()
                    break
        except:
            logger.error(
                f'Unexpected error in handle_self_play_client({client_data}):', exc_info=True)
            self._child_thread_error_flag.set()

    def handle_training_client(self, client_data: ClientData):
        try:
            while True:
                try:
                    msg = recv_json(client_data.sock)
                except OSError:
                    self.handle_disconnect(client_data)
                    return

                msg_type = msg['type']
                if msg_type == 'train-step-done':
                    self.handle_train_step_done(msg)
                elif msg_type == 'lock-gpu':
                    self.handle_lock_gpu(msg, client_data)
        except:
            logger.error('Unexpected error in handle_training_client():', exc_info=True)
            self._child_thread_error_flag.set()

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

    def flush_pending_games(self, client_id, commit_to_db=True):
        """
        Flushes pending games to the database, and returns the number of augmented positions.
        """
        if not self._pending_game_data:
            return 0

        db_conn = self.my_db_conn
        c = db_conn.cursor()

        n_augmented_positions = 0
        game_dict = defaultdict(list)  # keyed by gen
        for client_id, gen, start_timestamp, end_timestamp, n_rows in self._pending_game_data:
            n_augmented_positions += n_rows
            game_dict[gen].append((client_id, start_timestamp, end_timestamp, n_rows))

        for gen, game_data in game_dict.items():
            self._flush_pending_games_helper(c, gen, game_data)

        c.executemany('INSERT INTO games (client_id, gen, start_timestamp, end_timestamp, augmented_positions) VALUES (?, ?, ?, ?, ?)',
                      self._pending_game_data)
        self._pending_game_data = []

        if commit_to_db:
            db_conn.commit()
            self._increment_master_list_length(n_augmented_positions)

        return n_augmented_positions

    def _increment_master_list_length(self, n: int):
        with self._train_ready_lock:
            self._master_list_length += n
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                self._train_ready_event.set()

    def insert_metrics(self, client_id, gen, timestamp, metrics, commit_to_db=True):
        db_conn = self.my_db_conn
        c = db_conn.cursor()

        cache_hits = metrics['cache_hits']
        cache_misses = metrics['cache_misses']
        positions_evaluated = metrics['positions_evaluated']
        batches_evaluated = metrics['batches_evaluated']
        full_batches_evaluated = metrics['full_batches_evaluated']

        c.execute(
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

        c.execute("""UPDATE self_play_metadata
            SET positions_evaluated = positions_evaluated + ?,
                batches_evaluated = batches_evaluated + ?
            WHERE gen = ?""", (positions_evaluated, batches_evaluated, gen))

        if commit_to_db:
            db_conn.commit()

    def handle_metrics(self, msg, client_data: ClientData):
        client_id = client_data.client_id
        gen = msg['gen']
        timestamp = msg['timestamp']
        metrics = msg['metrics']

        n_augmented_positions = self.flush_pending_games(client_id, commit_to_db=False)
        self.insert_metrics(client_id, gen, timestamp, metrics, commit_to_db=False)
        self.my_db_conn.commit()
        self._increment_master_list_length(n_augmented_positions)

    def handle_game(self, msg, client_data: ClientData):
        client_id = client_data.client_id
        gen = msg['gen']
        start_timestamp = msg['start_timestamp']
        end_timestamp = msg['end_timestamp']
        rows = msg['rows']
        flush = msg['flush']

        self._pending_game_data.append((client_id, gen, start_timestamp, end_timestamp, rows))

        if flush:
            metrics = msg.get('metrics', None)

            n_augmented_positions = self.flush_pending_games(client_id, commit_to_db=False)
            if metrics:
                self.insert_metrics(
                    client_id, gen, end_timestamp, metrics, commit_to_db=False)
            self.my_db_conn.commit()
            self._increment_master_list_length(n_augmented_positions)

    def pause(self, clients: List[ClientData]):
        if not clients:
            return
        logger.info('Issuing pause...')
        data = {'type': 'pause'}

        for client in clients:
            send_json(client.sock, data)

        for client in clients:
            event = self._pause_ack_events[client.client_id]
            event.wait()
            event.clear()
        logger.info('Pause acked!')

    def get_client_data_list(self, ctype: ClientType) -> List[ClientData]:
        """
        Returns a list of all client datas of the given type.
        """
        with self._client_data_lock:
            return [c for c in self._client_data_list if c.client_type == ctype]

    def get_single_client_data(self, ctype: ClientType) -> ClientData:
        """
        Returns a single client data of the given type.
        """
        data_list = self.get_client_data_list(ctype)
        assert len(data_list) > 0, f'No clients of type {ctype} connected'
        return data_list[0]

    def pause_shared_gpu_self_play_clients(self, training_client_data: ClientData):
        self_play_list = self.get_client_data_list(ClientType.SELF_PLAY)
        shared_list = [c for c in self_play_list if c.shares_gpu_with(training_client_data)]
        self.pause(shared_list)

    def reload_weights(self, generation: int):
        clients = self.get_client_data_list(ClientType.SELF_PLAY)
        if not clients:
            return

        model_filename = self.organizer.get_model_filename(generation)
        logger.info('Issuing reload...')

        data = {
            'type': 'reload_weights',
            'model_filename': model_filename,
            'generation': generation,
            }

        for client in clients:
            send_json(client.sock, data)
