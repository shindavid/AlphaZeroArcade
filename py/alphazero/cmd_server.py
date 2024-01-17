from alphazero.sample_window_logic import SamplingParams, Window, construct_window, \
    get_required_dataset_size
from alphazero.custom_types import Generation
from alphazero.data.games_dataset import PositionDataset, PositionListSlice
from alphazero.net_trainer import TrainingStats
from util.py_util import timed_print
from util.socket_util import recvall

from collections import defaultdict
from dataclasses import dataclass
import json
import os
import socket
import sqlite3
import threading
import traceback
from typing import Dict, List, Optional


ClientId = int
ThreadId = int


def _get_next_json_msg(sock: socket.socket, timeout: Optional[float]=None):
    """
    Returns a json message from the socket.

    Raises an exception if the socket is closed. See recvall() for details on possible exceptions.
    """
    data = recvall(sock, 4, timeout=timeout)
    length = int.from_bytes(data, byteorder='big')

    data = recvall(sock, length, timeout=timeout)
    msg = json.loads(data.decode())
    return msg


@dataclass
class CmdServerClient:
    def __init__(self,
                 client_id: ClientId,
                 sock: socket.socket,
                 ip_address: str,
                 port: int,
                 proc_start_timestamp: int,
                 shared_gpu: bool):
        self.client_id = client_id
        self.ip_address = ip_address
        self.port = port
        self.proc_start_timestamp = proc_start_timestamp
        self.shared_gpu = shared_gpu

        self._sock_mutex = threading.Lock()
        self._sock_valid = True
        self._sock = sock

        self.pause_ack_event = threading.Event()

    def __str__(self):
        return f'CmdServerClient({self.client_id} ({self.ip_address}:{self.port})'

    def shutdown(self):
        with self._sock_mutex:
            self._sock_valid = False
            self._sock.close()

    def send_json(self, data) -> bool:
        """
        Sends data to the server.

        Returns True if the data was sent successfully, False otherwise. A failure should
        correspond to a client disconnection.

        Caller should decom the client if this returns False.
        """
        try:
            msg = json.dumps(data).encode()
            length = len(msg)
            with self._sock_mutex:
                if not self._sock_valid:
                    return False

                self._sock.sendall(length.to_bytes(4, byteorder='big'))
                self._sock.sendall(msg)
            return True
        except:
            print(f'CmdServerClient {self}) disconnected - send_json() failure')
            traceback.print_exc()
            self.shutdown()
            return False

    def get_next_json_msg(self):
        """
        Returns a json message from the client.

        In case of error or if the socket is already closed, shuts down and returns None.
        """
        try:
            with self._sock_mutex:
                if not self._sock_valid:
                    return None
                return _get_next_json_msg(self._sock)
        except ConnectionError:
            return None


class CmdServer:
    """
    A server that accepts connections from multiple c++ CmdServerClients and communicates with them.

    A new thread is spawned for each c++ client. The thread handles all communication with the
    client, writing data received from the client to a database.

    TODO: currently this class is responsible both for client-communications and for parts of the
    sampling window management logic. These responsibilities should be separated.
    """
    def __init__(self, sampling_params: SamplingParams, base_dir: str,
                 host='localhost', port=12345):
        self.base_dir = base_dir
        self.db_filename = os.path.join(base_dir, 'training.db')
        self.host = host
        self.port = port

        self._clients: Dict[ClientId, CmdServerClient] = {}
        self._clients_lock = threading.Lock()

        self._db_conn_dict: Dict[ThreadId, sqlite3.Connection] = {}
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()

        self._client_disconnect_event = threading.Event()
        self._pending_game_data = []

        self.sampling_params = sampling_params
        self.last_sample_window: Window = self.load_last_sample_window()
        self.master_list = PositionListSlice()  # lazily initialized

        # The length of the master_list can be computed on-demand by reading the database. To
        # avoid doing this repeatedly, we grab the value once at start-up, store it as a member, and
        # then update it manually whenever we add new games to the database.
        self._master_list_length = self.compute_master_list_length()

        # This is the length that the master_list needs to be before we can start a new train loop.
        # Initialized lazily.
        self._master_list_length_for_next_train_loop = 0

        self._train_ready_event = threading.Event()

        # Controls access to self._train_ready_event and self._master_list_length*
        self._train_ready_lock = threading.Lock()

        # self.dataset_generator = PositionDatasetGenerator(base_dir, self.db_conn, sampling_params)

    def get_position_dataset(self) -> PositionDataset:
        self.master_list.extend(self.my_db_conn.cursor())

        f = self.sampling_params.window_size_function
        n = self.master_list.end_index
        c = int(n - f(n))

        self.master_list.set_start_index(c)
        return PositionDataset(self.base_dir, self.master_list)

    def wait_until_enough_training_data(self):
        with self._train_ready_lock:
            self._master_list_length_for_next_train_loop = get_required_dataset_size(
                self.sampling_params, self.last_sample_window)
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

    @property
    def n_samples_per_window(self):
        return self.sampling_params.minibatch_size * self.sampling_params.minibatches_per_window

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
        # timed_print(f'Found {n_augmented_positions} positions in generation 0 (n_samples_per_window={self.n_samples_per_window})')
        return n_augmented_positions >= self.n_samples_per_window

    def load_last_sample_window(self) -> Window:
        cursor = self.my_db_conn.cursor()
        cursor.execute("""SELECT window_start, window_end, window_sample_rate
                          FROM training ORDER BY gen DESC LIMIT 1""")
        row = cursor.fetchone()
        if row is None:
            # kZero-style initialization of sample window
            params = self.sampling_params
            return Window(0, params.samples_per_window(), params.target_sample_rate)
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

    def get_clients_list(self) -> List[CmdServerClient]:
        with self._clients_lock:
            return list(self._clients.values())

    def wait_for_client_disconnect(self):
        self._client_disconnect_event.wait()
        self._client_disconnect_event.clear()

    def _create_db_conn(self):
        if os.path.isfile(self.db_filename):
            return sqlite3.connect(self.db_filename)

        conn = sqlite3.connect(self.db_filename)
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
            client_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            port INTEGER,
            proc_start_timestamp INTEGER
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
        with self._clients_lock:
            if client_id in self._clients:
                del self._clients[client_id]

    def accept_clients(self):
        """
        Loop that checks for new clients. For each new client, spawns a thread to handle it.
        """
        db_conn = self.my_db_conn
        while True:
            client_id = None
            try:
                client_socket, addr = self.server_socket.accept()
                ip_address, port = addr

                msg = _get_next_json_msg(client_socket, timeout=1)
                assert msg['type'] == 'handshake', f'Expected handshake from client, got {msg}'
                proc_start_timestamp = msg['proc_start_timestamp']
                shared_gpu = msg['shared_gpu']
                assert isinstance(proc_start_timestamp, int)

                c = db_conn.cursor()
                c.execute('INSERT INTO clients (ip_address, port, proc_start_timestamp) VALUES (?, ?, ?)',
                        (ip_address, port, proc_start_timestamp)
                        )
                client_id = c.lastrowid
                db_conn.commit()

                client = CmdServerClient(
                    client_id=client_id,
                    sock=client_socket,
                    ip_address=ip_address,
                    port=port,
                    proc_start_timestamp=proc_start_timestamp,
                    shared_gpu=shared_gpu,
                    )

                self._clients_lock.acquire()
                self._clients[client_id] = client
                self._clients_lock.release()

                timed_print(f'Accepted client {client_id} (shared_gpu={shared_gpu})')

                reply = {'type': 'handshake_ack', 'client_id': client_id}
                if not client.send_json(reply):
                    raise Exception(f'Failed to send handshake_ack to {client}')
                threading.Thread(target=self.handle_client, args=(client,), daemon=True).start()
            except:
                print('Exception in accept_clients()')
                traceback.print_exc()
                if client_id is not None:
                    self.remove_client(client_id)

    def handle_client(self, client: CmdServerClient):
        try:
            while True:
                msg = client.get_next_json_msg()
                if msg is None:
                    break

                msg_type = msg['type']
                if msg_type == 'pause_ack':
                    client.pause_ack_event.set()
                elif msg_type == 'metrics':
                    self.handle_metrics(msg, client)
                elif msg_type == 'game':
                    self.handle_game(msg, client)
        except:
            print(f'Exception in handle_client({client})')
            traceback.print_exc()
            print('')
        finally:
            timed_print(f'Closing socket for {client}')
            client.shutdown()
            self.remove_client(client.client_id)
            self.close_my_db_conn()
            self._client_disconnect_event.set()

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

    def handle_metrics(self, msg, client: CmdServerClient):
        gen = msg['gen']
        timestamp = msg['timestamp']
        metrics = msg['metrics']

        n_augmented_positions = self.flush_pending_games(client.client_id, commit_to_db=False)
        self.insert_metrics(client.client_id, gen, timestamp, metrics, commit_to_db=False)
        self.my_db_conn.commit()
        self._increment_master_list_length(n_augmented_positions)

    def handle_game(self, msg, client: CmdServerClient):
        client_id = client.client_id
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

    def pause(self, clients: List[CmdServerClient]):
        if not clients:
            return
        timed_print('Issuing pause...')
        data = {'type': 'pause'}

        for client in clients:
            client.send_json(data)

        for client in clients:
            client.pause_ack_event.wait()
            client.pause_ack_event.clear()
        timed_print('Pause acked!')

    def get_shared_gpu_clients(self) -> List[CmdServerClient]:
        """
        Returns a list of all clients that are sharing a GPU with the training process
        """
        return [c for c in self.get_clients_list() if c.shared_gpu]

    def pause_shared_gpu_clients(self):
        self.pause(self.get_shared_gpu_clients())

    def reload_weights(self, model_filename: str, generation: int):
        timed_print('Issuing reload...')
        clients = self.get_clients_list()

        data = {
            'type': 'reload_weights',
            'model_filename': model_filename,
            'generation': generation,
            }

        for client in clients:
            client.send_json(data)

    def record_training_step(self, stats: TrainingStats):
        window = construct_window(self.last_sample_window, stats.window_start, stats.window_end,
                                  stats.n_samples)
        self.last_sample_window = window

        c = self.my_db_conn.cursor()

        c.execute("""INSERT OR REPLACE INTO training (gen, training_start_ts, training_end_ts,
            window_start, window_end, window_sample_rate)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (stats.gen, stats.start_ts, stats.end_ts, window.start, window.end, window.sample_rate))

        head_data = []
        for head_stats in stats.substats_list:
            head_data.append((stats.gen, head_stats.name, head_stats.loss(),
                head_stats.loss_weight, head_stats.accuracy()))

        c.executemany("""INSERT OR REPLACE INTO training_heads (gen, head_name, loss, loss_weight, accuracy)
            VALUES (?, ?, ?, ?, ?)""", head_data)

        self.my_db_conn.commit()

    def start(self):
        threading.Thread(target=self.accept_clients, daemon=True).start()
