from alphazero.logic.common_params import CommonParams
from alphazero.logic import constants
from alphazero.logic.directory_organizer import DirectoryOrganizer
from game_index import get_game_spec
from util.logging_util import get_logger
from util.py_util import sha256sum
from util.repo_util import Repo
from util.socket_util import send_json, recv_json
from util import subprocess_util

from dataclasses import dataclass
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional


logger = get_logger()


@dataclass
class SelfPlayServerParams:
    training_server_host: str = 'localhost'
    training_server_port: int = constants.DEFAULT_TRAINING_SERVER_PORT
    binary_path: str = None
    cuda_device: str = 'cuda:0'

    @staticmethod
    def create(args) -> 'SelfPlayServerParams':
        return SelfPlayServerParams(
            training_server_host=args.training_server_host,
            training_server_port=args.training_server_port,
            binary_path=args.binary_path,
            cuda_device=args.cuda_device,
        )

    @staticmethod
    def add_args(parser):
        defaults = SelfPlayServerParams()
        group = parser.add_argument_group('SelfPlayServer options')

        group.add_argument('--training-server-host', type=str,
                           default=defaults.training_server_host,
                           help='training-server host (default: %(default)s)')
        group.add_argument('--training-server-port', type=int,
                           default=defaults.training_server_port,
                           help='training-server port (default: %(default)s)')
        group.add_argument('-b', '--binary-path',
                           help='binary path. By default, if a unique binary is found in the '
                           'alphazero dir, it will be used. If no binary is found in the alphazero '
                           'dir, then will use one found in REPO_ROOT/target/Release/bin/. If '
                           'multiple binaries are found in the alphazero dir, then this option is '
                           'required.')
        group.add_argument('--cuda-device', default=defaults.cuda_device,
                           help='cuda device (default: %(default)s)')



class SelfPlayServer:
    def __init__(self, params: SelfPlayServerParams, common_params: CommonParams):
        self.organizer = DirectoryOrganizer(common_params)
        self.game_spec = get_game_spec(common_params.game)
        self.training_server_host = params.training_server_host
        self.training_server_port = params.training_server_port
        self.cuda_device = params.cuda_device
        self.training_server_socket = None

        self.child_process = None
        self.client_id = None

        self._shutdown_code = None

        self._binary_path_set = False
        self._binary_path = params.binary_path

    def register_signal_handler(self):
        def signal_handler(sig, frame):
            logger.info('Detected Ctrl-C.')
            self.shutdown(0)

        signal.signal(signal.SIGINT, signal_handler)

    def __str__(self):
        client_id_str = '???' if self.client_id is None else str(self.client_id)
        return f'SelfPlayServer({client_id_str})'

    @property
    def bins_dir(self):
        return self.organizer.bins_dir

    def copy_extras(self):
        for extra in self.game_spec.extra_runtime_deps:
            extra_src = os.path.join(Repo.root(), extra)
            extra_tgt = os.path.join(self.bins_dir, 'extra', os.path.basename(extra))
            rsync_cmd = ['rsync', '-t', extra_src, extra_tgt]
            subprocess_util.run(rsync_cmd)

    def copy_binary(self, bin_src):
        bin_md5 = str(sha256sum(bin_src))
        bin_tgt = os.path.join(self.bins_dir, bin_md5)
        rsync_cmd = ['rsync', '-t', bin_src, bin_tgt]
        subprocess_util.run(rsync_cmd)
        return bin_tgt

    @property
    def binary_path(self):
        """
        TODO: perhaps during the handshake with the training-server, the training-server should send
        the binary, and the self-play server should run that. This would reduce operational overhead
        by putting the onus of building and distributing the binary in one spot (the training
        server).
        """
        if self._binary_path_set:
            return self._binary_path

        self._binary_path_set = True
        if self._binary_path:
            bin_tgt = self.copy_binary(self._binary_path)
            self.copy_extras()
            logger.info(
                f'Using cmdline-specified binary {self._binary_path} (copied to {bin_tgt})')
            self._binary_path = bin_tgt
            return self._binary_path

        bin_tgt = self.organizer.get_latest_binary()
        if bin_tgt is None:
            bin_name = self.game_spec.name
            bin_src = os.path.join(
                Repo.root(), f'target/Release/bin/{bin_name}')
            bin_tgt = self.copy_binary(bin_src)
            self.copy_extras()
            self._binary_path = bin_tgt
            logger.info(f'Using binary {bin_src} (copied to {bin_tgt})')
        else:
            self._binary_path = bin_tgt
            logger.info(f'Using most-recently used binary: {bin_tgt}')
            self.copy_extras()

        return self._binary_path

    def run(self):
        training_server_address = (self.training_server_host, self.training_server_port)
        training_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        training_server_socket.connect(training_server_address)

        self.training_server_socket = training_server_socket
        self.send_handshake()
        self.recv_handshake()

        threading.Thread(target=self.recv_loop, daemon=True).start()
        self.main_loop()

    def main_loop(self):
        while True:
            time.sleep(1)
            if self.child_process is not None and self.child_process.poll() is not None:
                if self.child_process.returncode != 0:
                    logger.error(f'Child process exited with code {self.child_process.returncode}')
                    self._shutdown_code = 1

            if self._shutdown_code is not None:
                self.shutdown(self._shutdown_code)
                break

    def send_handshake(self):
        data = {
            'type': 'handshake',
            'role': 'self-play-wrapper',
            'start_timestamp': time.time_ns(),
            }

        send_json(self.training_server_socket, data)

    def recv_handshake(self):
        data = recv_json(self.training_server_socket, timeout=1)
        assert data['type'] == 'handshake_ack', data

        self.client_id = data['client_id']
        logger.info(f'Received client id assignment: {self.client_id}')

    def recv_loop(self):
        try:
            while True:
                msg = recv_json(self.training_server_socket)

                msg_type = msg['type']
                if msg_type == 'start-gen0':
                    self.start_gen0(msg)
                elif msg_type == 'start':
                    self.start(msg)
                elif msg_type == 'quit':
                    self.quit()
                    break
        except ConnectionError as e:
            if str(e).find('Socket gracefully closed by peer') != -1:
                logger.info(f'Socket gracefully closed by peer')
                self._shutdown_code = 0
                return
            else:
                logger.error(
                    f'Unexpected error in recv_loop():', exc_info=True)
                self._shutdown_code = 1
                return
        except:
            logger.error(f'Unexpected error in recv_loop():', exc_info=True)
            self._shutdown_code = 1
            return

    def start_gen0(self, msg):
        assert self.child_process is None

        # TODO: once we change c++ to directly communicate game data to the training-server via TCP,
        # we will no longer need games_base_dir here
        games_base_dir = msg['games_base_dir']
        max_rows = msg['max_rows']
        gen = 0

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--games-base-dir', games_base_dir,
            '--do-not-report-metrics',
            '--max-rows', max_rows,

            # for gen-0, sample more positions and use fewer iters per game, so we finish faster
            '--num-full-iters', 100,
            '--full-pct', 1.0,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--training-server-hostname', self.training_server_host,
            '--training-server-port', self.training_server_port,
            '--starting-generation', gen,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play.log')
        with open(log_filename, 'a') as log_file:
            logger.info(f'Running gen-0 self-play: {self_play_cmd}')
            subprocess_util.run(self_play_cmd, stdout=log_file,
                                stderr=log_file, check=True)
            logger.info(f'Gen-0 self-play complete!')

    def start(self, msg):
        assert self.child_process is None

        # TODO: once we change c++ to directly communicate game data to the training-server via TCP,
        # we will no longer need games_base_dir or model here
        games_base_dir = msg['games_base_dir']
        gen = msg['gen']
        model = msg['model']

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--games-base-dir', games_base_dir,
            '-m', model,
            '--cuda-device', self.cuda_device,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--training-server-hostname', self.training_server_host,
            '--training-server-port', self.training_server_port,
            '--starting-generation', gen,
            '--cuda-device', self.cuda_device,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play.log')
        log_file = open(log_filename, 'a')
        proc = subprocess_util.Popen(self_play_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        self.child_process = proc
        logger.info(f'Running self-play [{proc.pid}]: {self_play_cmd}')

    def quit(self):
        logger.info(f'Received quit command')
        self._shutdown_code = 0

    def shutdown(self, code):
        logger.info(f'Shutting down...')
        if self.training_server_socket:
            self.training_server_socket.close()
        sys.exit(code)
