from alphazero.common_args import CommonArgs
from alphazero.custom_types import Generation
from alphazero.data.position_dataset import PositionDataset, PositionListSlice
from alphazero.directory_organizer import DirectoryOrganizer, PathInfo
from alphazero.net_trainer import NetTrainer
from alphazero.training_params import TrainingParams
from games import get_game_type
from net_modules import Model
from util.logging_util import get_logger
from util.py_util import make_hidden_filename
from util.socket_util import send_json, recv_json

import os
import shutil
import signal
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Optional, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader


logger = get_logger()


class TrainingServer:
    """
    NOTE: the separation of this class from CmdServer gives the impression that the two servers can
    be run on different machines. However, the reality is that they are coupled via their
    dependence on the same filesystem directory. The training server reads data files from the
    directory and writes model/checkpoint files to the directory. In the future, we may want to
    decouple by having the cmd server do the filesystem reads/writes, with the training server and
    cmd server communicating to each other via TCP.
    """
    def __init__(self, cmd_server_host: str, cmd_server_port: int, cuda_device: str,
                 model_cfg: str):
        self.organizer = DirectoryOrganizer()
        self.cmd_server_host = cmd_server_host
        self.cmd_server_port = cmd_server_port
        self.cuda_device = cuda_device
        self.game_type = get_game_type(CommonArgs.game)
        self.model_cfg = model_cfg

        self.cmd_server_socket = None
        self.client_id = None

        self._shutdown_code = None
        self._child_thread_error_flag = threading.Event()

        self._net = None
        self._opt = None
        self._base_dir = None
        self._db_filename = None
        self._db_conn_dict = {}
        self._master_list = PositionListSlice()

    @property
    def my_db_conn(self) -> sqlite3.Connection:
        """
        sqlite3 demands a single connection per thread. This property hides this detail under the
        hood.
        """
        thread_id = threading.get_ident()
        conn = self._db_conn_dict.get(thread_id, None)
        if conn is None:
            conn = sqlite3.connect(self._db_filename)
            self._db_conn_dict[thread_id] = conn
        return conn

    def register_signal_handler(self):
        def signal_handler(sig, frame):
            logger.info('Detected Ctrl-C.')
            self.shutdown(0)

        signal.signal(signal.SIGINT, signal_handler)

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

    def run(self):
        cmd_server_address = (self.cmd_server_host, self.cmd_server_port)
        cmd_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cmd_server_socket.connect(cmd_server_address)

        self.cmd_server_socket = cmd_server_socket
        self.send_handshake()
        self.recv_handshake()

        threading.Thread(target=self.recv_loop, daemon=True).start()
        self.main_loop()

    def main_loop(self):
        while True:
            time.sleep(1)
            if self._shutdown_code is not None:
                self.shutdown(self._shutdown_code)
                break

    def send_handshake(self):
        data = {
            'type': 'handshake',
            'role': 'training',
            'start_timestamp': time.time_ns(),
        }

        send_json(self.cmd_server_socket, data)

    def recv_handshake(self):
        data = recv_json(self.cmd_server_socket, timeout=1)
        assert data['type'] == 'handshake_ack', data

        self.client_id = data['client_id']
        self._base_dir = data['base_dir']

        db_filename = data['db_filename']
        assert os.path.isfile(db_filename), db_filename
        self._db_filename = db_filename

        logger.info(f'Received client id assignment: {self.client_id}')

    def recv_loop(self):
        try:
            while True:
                msg = recv_json(self.cmd_server_socket)

                msg_type = msg['type']
                if msg_type == 'train-step':
                    self.train_step(msg)
                elif msg_type == 'quit':
                    self.quit()
                    break
        except ConnectionError as e:
            if str(e).find('Socket gracefully closed by peer') != -1:
                logger.info(f'Socket gracefully closed by peer')
                self._shutdown_code = 0
                return
            else:
                logger.error(f'Unexpected error in recv_loop():', exc_info=True)
                self._shutdown_code = 1
                return
        except:
            logger.error(f'Unexpected error in recv_loop():', exc_info=True)
            self._shutdown_code = 1

    def train_step(self, msg):
        gen = msg['gen']
        start = msg['start']
        end = msg['end']
        n_minibatches = msg['n_minibatches']
        minibatch_size = msg['minibatch_size']

        cursor = self.my_db_conn.cursor()
        self._master_list.set_bounds(cursor, start, end)
        dataset = PositionDataset(self._base_dir, self._master_list)

        logger.info('******************************')
        logger.info(f'Train gen:{gen}')
        dataset.announce_sampling(logger.info)

        trainer = NetTrainer(gen, n_minibatches, self.cuda_device)

        loader = DataLoader(
            dataset,
            batch_size=minibatch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True)

        net, optimizer = self.get_net_and_optimizer(loader)

        data = {
            'type': 'lock-gpu',
        }
        send_json(self.cmd_server_socket, data)
        msg = recv_json(self.cmd_server_socket)
        assert msg['type'] == 'lock-gpu-ack', msg

        stats = trainer.do_training_epoch(loader, net, optimizer, dataset)
        stats.dump(logger.info)

        assert trainer.n_minibatches_processed >= n_minibatches

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

        data = {
            'type': 'train-step-done',
            'stats': stats.to_json(),
        }
        send_json(self.cmd_server_socket, data)

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
        self._net.cuda(device=self.cuda_device)
        self._net.train()

        learning_rate = TrainingParams.learning_rate
        momentum = TrainingParams.momentum
        weight_decay = TrainingParams.weight_decay
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
                self.game_type.model_dict[self.model_cfg](input_shape))
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
        logger.info(f'Shutting down...')
        if self.cmd_server_socket:
            self.cmd_server_socket.close()
        sys.exit(code)
