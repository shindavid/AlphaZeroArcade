from alphazero.data.position_dataset import PositionDataset, PositionListSlice
from alphazero.logic.common_params import CommonParams
from alphazero.logic import constants
from alphazero.logic.custom_types import ChildThreadError, Generation
from alphazero.logic.directory_organizer import DirectoryOrganizer, PathInfo
from alphazero.logic.learning_params import LearningParams
from alphazero.logic.net_trainer import NetTrainer
from alphazero.logic.sample_window_logic import SamplingParams, Window, construct_window, \
    get_required_dataset_size
from game_index import get_game_spec
from net_modules import Model
from util.logging_util import get_logger
from util.py_util import make_hidden_filename
from util.socket_util import send_json, recv_json
from util.sqlite3_util import ConnectionPool

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import os
import shutil
import signal
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader


logger = get_logger()


@dataclass
class TrainingServerParams:
    port: int = constants.DEFAULT_TRAINING_SERVER_PORT
    cuda_device: str = 'cuda:0'
    model_cfg: str = 'default'

    @staticmethod
    def create(args) -> 'TrainingServerParams':
        return TrainingServerParams(
            port=args.port,
            cuda_device=args.cuda_device,
            model_cfg=args.model_cfg,
        )

    @staticmethod
    def add_args(parser):
        defaults = TrainingServerParams()
        group = parser.add_argument_group('TrainingServer options')

        group.add_argument('--port', type=int,
                           default=defaults.port,
                           help='TrainingServer port (default: %(default)s)')
        group.add_argument('--cuda-device',
                           default=defaults.cuda_device,
                           help='cuda device used for network training (default: %(default)s)')
        group.add_argument('-m', '--model-cfg', default=defaults.model_cfg,
                           help='model config (default: %(default)s)')

    def add_to_cmd(self, cmd: List[str]):
        defaults = TrainingServerParams()
        if self.port != defaults.port:
            cmd.extend(['--port', str(self.port)])
        if self.cuda_device != defaults.cuda_device:
            cmd.extend(['--cuda-device', self.cuda_device])
        if self.model_cfg != defaults.model_cfg:
            cmd.extend(['--model-cfg', self.model_cfg])


ClientId = int
ThreadId = int


class ClientType(Enum):
    SELF_PLAY_WRAPPER = 'self-play-wrapper'
    SELF_PLAY = 'self-play'


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

    def is_on_localhost(self):
        return self.ip_address == '127.0.0.1'

    def __str__(self):
        tokens = [str(self.client_type), str(self.client_id),
                  f'{self.ip_address}:{self.port}', self.cuda_device]
        tokens = [t for t in tokens if t]
        return f'ClientData({", ".join(tokens)})'


class TrainingServer:
    def __init__(self, params: TrainingServerParams, learning_params: LearningParams,
                 sampling_params: SamplingParams, common_params: CommonParams):
        self.organizer = DirectoryOrganizer(common_params)
        self.params = params
        self.game_spec = get_game_spec(common_params.game)
        self.learning_params = learning_params
        self.sampling_params = sampling_params

        self._pause_ack_events: Dict[ClientId, threading.Event] = {}
        self._client_data_list: List[ClientData] = []
        self._client_data_lock = threading.Lock()
        self._pending_game_data = []

        self._server_socket = None
        self._last_sample_window = None
        self._master_list_length = None
        self._master_list_length_for_next_train_loop = None
        self._master_list = PositionListSlice()

        self._shutdown_code = None
        self._child_thread_error_flag = threading.Event()

        self._train_ready_event = threading.Event()
        self._train_ready_lock = threading.Lock()

        self._self_play_done_event = threading.Event()
        self._self_play_client_connected = False
        self._self_play_client_connected_event = threading.Event()

        self._net = None
        self._opt = None

        self.clients_db_conn_pool = ConnectionPool(
            self.organizer.clients_db_filename, constants.CLIENTS_TABLE_CREATE_CMDS)
        self.training_db_conn_pool = ConnectionPool(
            self.organizer.training_db_filename, constants.TRAINING_TABLE_CREATE_CMDS)
        self.self_play_db_conn_pool = ConnectionPool(
            self.organizer.self_play_db_filename, constants.SELF_PLAY_TABLE_CREATE_CMDS)

    def register_signal_handler(self):
        def signal_handler(sig, frame):
            logger.info(f'Detected Ctrl-C in thread {threading.current_thread().name}.')
            self.shutdown(0)

        signal.signal(signal.SIGINT, signal_handler)

    @property
    def model_cfg(self):
        return self.params.model_cfg

    def __str__(self):
        client_id_str = '???' if self.client_id is None else str(
            self.client_id)
        return f'TrainingServer({client_id_str})'

    def get_latest_checkpoint_info(self) -> Optional[PathInfo]:
        return DirectoryOrganizer.get_latest_info(self.organizer.checkpoints_dir)

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self.organizer.checkpoints_dir, f'gen-{gen}.pt')

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.organizer.models_dir, f'gen-{gen}.pt')

    def load_last_sample_window(self) -> Window:
        cursor = self.training_db_conn_pool.get_cursor()
        cursor.execute("""SELECT window_start, window_end, window_sample_rate
                          FROM training ORDER BY gen DESC LIMIT 1""")
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            # kZero-style initialization of sample window
            samples_per_window = self.sampling_params.samples_per_window()
            target_sample_rate = self.sampling_params.target_sample_rate
            return Window(0, samples_per_window, target_sample_rate)
        return Window(*row)

    def compute_master_list_length(self) -> int:
        # Return cumulative_augmented_positions for the last row of games:
        cursor = self.self_play_db_conn_pool.get_cursor()
        cursor.execute("""SELECT cumulative_augmented_positions
                       FROM games ORDER BY id DESC LIMIT 1""")
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return 0
        return row[0]

    def _increment_master_list_length(self, n: int):
        with self._train_ready_lock:
            self._master_list_length += n
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                self._train_ready_event.set()

    def launch_self_play(self):
        gen = self.organizer.get_latest_model_generation()
        model_filename = self.organizer.get_model_filename(gen)

        data = {
            'type': 'start',
            'gen': gen,
            'games_base_dir': self.organizer.self_play_data_dir,
            'model': model_filename,
        }

        self.wait_for_self_play_client_connection()

        for client_data in self.get_client_data_list(ClientType.SELF_PLAY_WRAPPER):
            logger.info(f'Requesting {client_data} to launch self-play...')
            send_json(client_data.sock, data)

    def wait_until_enough_training_data(self):
        with self._train_ready_lock:
            self._master_list_length_for_next_train_loop = get_required_dataset_size(
                self.sampling_params, self._last_sample_window)
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                return
            self._train_ready_event.clear()

        logger.info('Waiting for more training data...')
        self._train_ready_event.wait()

    def wait_for_self_play_client_connection(self):
        if self._self_play_client_connected_event.is_set():
            return

        logger.info('Waiting for self play client connection...')
        while True:
            self._self_play_client_connected_event.wait(timeout=1)
            if self._self_play_client_connected_event.is_set():
                return

            if self._child_thread_error_flag.is_set():
                logger.info(
                    'Child thread error caught, exiting wait_for_self_play_client_connection() loop')
                raise ChildThreadError()

    def is_gen0_complete(self) -> bool:
        """
        Returns True if the first generation has been completed, False otherwise.
        """
        cursor = self.self_play_db_conn_pool.get_cursor()
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
        return n_augmented_positions >= self.sampling_params.samples_per_window()

    def run_gen0_if_necessary(self):
        if self.is_gen0_complete():
            return

        self.wait_for_self_play_client_connection()

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

    def insert_metrics(self, client_id, gen, timestamp, metrics, cursor=None):
        commit = not bool(cursor)
        if cursor is None:
            conn = self.self_play_db_conn_pool.get_connection(readonly=False)
            cursor = conn.cursor()

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

        if commit:
            cursor.connection.commit()
            cursor.close()

    def handle_metrics(self, msg, client_data: ClientData):
        client_id = client_data.client_id
        gen = msg['gen']
        timestamp = msg['timestamp']
        metrics = msg['metrics']

        conn = self.self_play_db_conn_pool.get_connection(readonly=False)
        cursor = conn.cursor()
        n_augmented_positions = self.flush_pending_games(client_id, cursor=cursor)
        self.insert_metrics(client_id, gen, timestamp, metrics, cursor=cursor)
        cursor.close()
        conn.commit()
        self._increment_master_list_length(n_augmented_positions)

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

            conn = self.self_play_db_conn_pool.get_connection(readonly=False)
            cursor = conn.cursor()
            n_augmented_positions = self.flush_pending_games(client_id, cursor=cursor)
            if metrics:
                self.insert_metrics(client_id, gen, end_timestamp, metrics, cursor=cursor)
            cursor.close()
            conn.commit()
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

    def pause_shared_gpu_self_play_clients(self):
        self_play_list = self.get_client_data_list(ClientType.SELF_PLAY)
        shared_list = [c for c in self_play_list if c.is_on_localhost() and
                       c.cuda_device == self.params.cuda_device]
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

    def accept_clients(self):
        """
        Loop that checks for new clients. For each new client, spawns a thread to handle it.
        """
        try:
            conn = self.clients_db_conn_pool.get_connection(readonly=False)
            while True:
                client_socket, addr = self._server_socket.accept()
                ip_address, port = addr

                msg = recv_json(client_socket)  # , timeout=1)
                assert msg['type'] == 'handshake', f'Expected handshake from client, got {msg}'
                role = msg['role']
                client_type = ClientType(role)

                reply = {'type': 'handshake_ack'}

                if client_type == ClientType.SELF_PLAY_WRAPPER:
                    handler_fn = self.handle_self_play_wrapper_client
                elif client_type == ClientType.SELF_PLAY:
                    handler_fn = self.handle_self_play_client
                else:
                    raise Exception(f'Unknown client type: {client_type}')

                start_timestamp = msg['start_timestamp']
                cuda_device = msg.get('cuda_device', '')

                cursor = conn.cursor()
                cursor.execute('INSERT INTO clients (ip_address, port, role, start_timestamp, cuda_device) VALUES (?, ?, ?, ?, ?)',
                          (ip_address, port, role, start_timestamp, cuda_device)
                          )
                client_id = cursor.lastrowid
                conn.commit()

                client_data = ClientData(
                    client_type, client_id, client_socket, start_timestamp, cuda_device)

                self._client_data_lock.acquire()
                self._client_data_list.append(client_data)
                self._client_data_lock.release()

                logger.info(f'Accepted client: {client_data}')

                reply['client_id'] = client_id
                send_json(client_socket, reply)
                name = f'handler-{str(client_type).split(".")[1]}-{client_id}'
                threading.Thread(target=handler_fn, name=name, args=(client_data,), daemon=True).start()

                self.setup_done_check()
        except:
            logger.error('Exception in accept_clients():', exc_info=True)
            self._child_thread_error_flag.set()

    def setup_done_check(self):
        with self._client_data_lock:
            types = set()
            for client_data in self._client_data_list:
                types.add(client_data.client_type)

            if not self._self_play_client_connected:
                if ClientType.SELF_PLAY_WRAPPER in types:
                    self._self_play_client_connected = True
                    self._self_play_client_connected_event.set()

    def remove_client(self, client_id: ClientId):
        with self._client_data_lock:
            self._client_data_list = [
                c for c in self._client_data_list if c.client_id != client_id]

    def handle_disconnect(self, client_data: ClientData):
        logger.info(f'Handling disconnect for {client_data}...')
        self.remove_client(client_data.client_id)
        self.close_my_db_conns()
        client_data.sock.close()
        logger.info(f'Disconnect complete!')

    def close_my_db_conns(self):
        thread_id = threading.get_ident()
        for pool in [self.clients_db_conn_pool, self.training_db_conn_pool, self.self_play_db_conn_pool]:
            pool.close_connections(thread_id)

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

    def flush_pending_games(self, client_id, cursor=None):
        """
        Flushes pending games to the database, and returns the number of augmented positions.

        Commits to the database unless cursor is provided, in which case the commit is left to the
        caller.
        """
        if not self._pending_game_data:
            return 0

        commit = not bool(cursor)
        if cursor is None:
            conn = self.self_play_db_conn_pool.get_connection(readonly=False)
            cursor = conn.cursor()

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

        if commit:
            cursor.connection.commit()
            self._increment_master_list_length(n_augmented_positions)
            cursor.close()

        return n_augmented_positions

    def _run_setup(self):
        logger.info('Performing TrainingServer setup...')
        self.organizer.makedirs()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.setblocking(True)
        self._server_socket.bind(('localhost', self.params.port))
        self._server_socket.listen()

        self._last_sample_window: Window = self.load_last_sample_window()

        # The length of the master_list can be computed on-demand by reading the database. To
        # avoid doing this repeatedly, we grab the value once at start-up, store it as a member, and
        # then update it manually whenever we add new games to the database.
        self._master_list_length = self.compute_master_list_length()

        # This is the length that the master_list needs to be before we can start a new train loop.
        # Initialized lazily.
        self._master_list_length_for_next_train_loop = 0

        logger.info(f'Listening for clients on port {self.params.port}...')
        threading.Thread(target=self.accept_clients, name='accept_clients', daemon=True).start()

    def run(self):
        try:
            threading.Thread(target=self.train_loop, name='train_loop', daemon=True).start()

            while True:
                if self._child_thread_error_flag.is_set():
                    logger.info('Child thread error detected, shutting down...')
                    self.shutdown(1)
                    return

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C, shutting down...')
            self.shutdown(0)
        except:
            logger.error('Unexpected error in run():', exc_info=True)
            logger.info('Shutting down...')
            self.shutdown(1)

    def train_loop(self):
        try:
            self._run_setup()
            self.run_gen0_if_necessary()
            self.train_gen1_model_if_necessary()
            self.launch_self_play()

            while True:
                self.wait_until_enough_training_data()
                self.launch_train_step()
                if self._child_thread_error_flag.is_set():
                    return
        except:
            logger.error('Unexpected error in train_loop():', exc_info=True)
            self._child_thread_error_flag.set()

    def train_gen1_model_if_necessary(self):
        gen = 1
        model_filename = self.organizer.get_model_filename(gen)
        if os.path.isfile(model_filename):
            return

        self.launch_train_step()

    def launch_train_step(self):
        """
        Launches a training step in a separate thread.

        Using a separate thread ensures that the DataLoader is properly cleaned up after the
        training step is complete.
        """
        thread = threading.Thread(target=self.train_step, name='train_step', daemon=True)
        thread.start()
        thread.join()

    def train_step(self):
        try:
            self.train_step_helper()
        except:
            logger.error('Unexpected error in train_step():', exc_info=True)
            self._child_thread_error_flag.set()

    def train_step_helper(self):
        gen = self.organizer.get_latest_model_generation() + 1

        cursor = self.self_play_db_conn_pool.get_cursor()
        cursor.execute(
            """SELECT cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1""")
        row = cursor.fetchone()
        n = row[0]

        f = self.sampling_params.window_size_function
        n = row[0]
        c = int(n - f(n))

        start = c
        end = n
        n_minibatches = self.sampling_params.minibatches_per_epoch
        minibatch_size = self.sampling_params.minibatch_size

        self._master_list.set_bounds(cursor, start, end)
        cursor.close()
        dataset = PositionDataset(self.organizer.base_dir, self._master_list)

        logger.info('******************************')
        logger.info(f'Train gen:{gen}')
        dataset.announce_sampling(logger.info)

        trainer = NetTrainer(gen, n_minibatches, self.params.cuda_device)

        def disable_signal(worker_id):
            """
            Without this, the main process AND the DataLoader worker processes all handle the
            SIGTERM, which is not desirable.

            Using this function as worker_init_fn in DataLoader seems to disable the SIGTERM
            handling in the worker processes.

            In theory, the main process should still handle SIGTERM. But for reasons I don't
            understand, using disable_signal() seems to sometimes disable SIGTERM handling in the
            main process.
            """
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        loader = DataLoader(
            dataset,
            batch_size=minibatch_size,
            num_workers=4,
            worker_init_fn=disable_signal,
            pin_memory=True,
            shuffle=True)

        net, optimizer = self.get_net_and_optimizer(loader)

        self.pause_shared_gpu_self_play_clients()

        stats = trainer.do_training_epoch(loader, net, optimizer, dataset)
        stats.dump(logger.info)
        assert stats.n_minibatches_processed >= n_minibatches

        logger.info(f'Gen {gen} training complete')
        trainer.dump_timing_stats(logger.info)

        checkpoint_filename = self.get_checkpoint_filename(gen)
        model_filename = self.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        checkpoint = {}
        net.add_to_checkpoint(checkpoint)
        torch.save(checkpoint, tmp_checkpoint_filename)
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        logger.info(f'Checkpoint saved: {checkpoint_filename}')
        logger.info(f'Model saved: {model_filename}')

        window_start = stats.window_start
        window_end = stats.window_end
        n_samples = stats.n_samples
        start_ts = stats.start_ts
        end_ts = stats.end_ts

        window = construct_window(
            self._last_sample_window, window_start, window_end, n_samples)
        self._last_sample_window = window

        conn = self.training_db_conn_pool.get_connection(readonly=False)
        cursor = conn.cursor()

        cursor.execute("""INSERT OR REPLACE INTO training (gen, training_start_ts, training_end_ts,
            minibatch_size, n_minibatches, window_start, window_end, window_sample_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (gen, start_ts, end_ts, minibatch_size, n_minibatches,
                   window.start, window.end, window.sample_rate))

        head_data = []
        for head_stats in stats.substats_list:
            head_name = head_stats.name
            loss = head_stats.loss()
            accuracy = head_stats.accuracy()
            loss_weight = head_stats.loss_weight
            head_data.append((gen, head_name, loss, loss_weight, accuracy))

        cursor.executemany("""INSERT OR REPLACE INTO training_heads (gen, head_name, loss, loss_weight, accuracy)
            VALUES (?, ?, ?, ?, ?)""", head_data)

        conn.commit()
        cursor.close()
        self.close_my_db_conns()
        self.reload_weights(gen)

    def load_last_checkpoint(self):
        """
        If a prior checkpoint exists, does the following:

        - Sets self._net
        - Sets self._opt
        - Sets self.dataset_generator.expected_sample_counts
        """
        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            return

        gen = checkpoint_info.generation
        checkpoint_filename = self.get_checkpoint_filename(gen)
        logger.info(f'Loading checkpoint: {checkpoint_filename}')

        # copying the checkpoint to somewhere local first seems to bypass some sort of
        # filesystem issue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.pt')
            shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
            checkpoint = torch.load(tmp_checkpoint_filename)
            self._net = Model.load_from_checkpoint(checkpoint)

        self._init_net_and_opt()

    def _init_net_and_opt(self):
        """
        Assumes that self._net has been initialized, and that self._opt has not.

        Moves self._net to cuda device and puts it in train mode.

        Initializes self._opt.
        """
        self._net.cuda(device=self.params.cuda_device)
        self._net.train()

        learning_rate = self.learning_params.learning_rate
        momentum = self.learning_params.momentum
        weight_decay = self.learning_params.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

    def get_net_and_optimizer(self, loader: DataLoader) -> Tuple[Model, optim.Optimizer]:
        if self._net is not None:
            return self._net, self._opt

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            input_shape = loader.dataset.get_input_shape()
            target_names = loader.dataset.get_target_names()
            self._net = Model(
                self.game_spec.model_configs[self.model_cfg](input_shape))
            self._net.validate_targets(target_names)
            self._init_net_and_opt()
            logger.info(f'Creating new net with input shape {input_shape}')
        else:
            self.load_last_checkpoint()

        return self._net, self._opt

    def quit(self):
        logger.info(f'Received quit command')
        self._shutdown_code = 0

    def shutdown(self, code):
        if self._server_socket:
            self._server_socket.close()
        logger.info(f'Shut down complete! (code: {code})')
        sys.exit(code)
