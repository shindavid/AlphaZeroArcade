from alphazero.common_args import CommonArgs
from alphazero.directory_organizer import DirectoryOrganizer
from games import get_game_type
from util.logging_util import get_logger
from util.py_util import sha256sum
from util.repo_util import Repo
from util.socket_util import send_json, recv_json
from util import subprocess_util

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional


logger = get_logger()


class SelfPlayServer:
    def __init__(self, cmd_server_host: str, cmd_server_port: int, cuda_device: str,
                 binary_path: Optional[str] = None):
        self.organizer = DirectoryOrganizer()
        self.game_type = get_game_type(CommonArgs.game)
        self.cmd_server_host = cmd_server_host
        self.cmd_server_port = cmd_server_port
        self.cuda_device = cuda_device
        self.cmd_server_socket = None

        self.child_process = None
        self.client_id = None

        self._shutdown_code = None

        self._binary_path_set = False
        self._binary_path = binary_path

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

    def copy_binary(self, bin_src):
        bin_md5 = str(sha256sum(bin_src))
        bin_tgt = os.path.join(self.bins_dir, bin_md5)
        rsync_cmd = ['rsync', '-t', bin_src, bin_tgt]
        subprocess_util.run(rsync_cmd)
        return bin_tgt

    @property
    def binary_path(self):
        """
        TODO: perhaps during the handshake with the cmd-server, the cmd-server should send the
        binary, and the self-play server should run that. This would reduce operational overhead
        by putting the onus of building and distributing the binary in one spot (the cmd server).
        """
        if self._binary_path_set:
            return self._binary_path

        self._binary_path_set = True
        if self._binary_path:
            bin_tgt = self.copy_binary(self._binary_path)
            logger.info(
                f'Using cmdline-specified binary {self._binary_path} (copied to {bin_tgt})')
            self._binary_path = bin_tgt
            return self._binary_path

        candidates = os.listdir(self.bins_dir)
        if len(candidates) == 0:
            bin_name = self.game_type.binary_name
            bin_src = os.path.join(
                Repo.root(), f'target/Release/bin/{bin_name}')
            bin_tgt = self.copy_binary(bin_src)
            self._binary_path = bin_tgt
            logger.info(f'Using binary {bin_src} (copied to {bin_tgt})')
        else:
            # get the candidate with the most recent mtime:
            candidates = [os.path.join(self.bins_dir, c) for c in candidates]
            candidates = [(os.path.getmtime(c), c) for c in candidates]
            candidates.sort()
            bin_tgt = candidates[-1][1]
            self._binary_path = bin_tgt
            logger.info(f'Using most-recently used binary: {bin_tgt}')

        return self._binary_path

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
            'role': 'self-play-wrapper',
            'start_timestamp': time.time_ns(),
            }

        send_json(self.cmd_server_socket, data)

    def recv_handshake(self):
        data = recv_json(self.cmd_server_socket, timeout=1)
        assert data['type'] == 'handshake_ack', data

        self.client_id = data['client_id']
        logger.info(f'Received client id assignment: {self.client_id}')

    def recv_loop(self):
        try:
            while True:
                msg = recv_json(self.cmd_server_socket)

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

        # TODO: once we change c++ to directly communicate game data to cmd-server via TCP, we will
        # no longer need games_base_dir here
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
            '--cmd-server-host', self.cmd_server_host,
            '--cmd-server-port', self.cmd_server_port,
            '--starting-generation', gen,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play-{self.client_id}-gen0.log')
        with open(log_filename, 'a') as log_file:
            logger.info(f'Running gen-0 self-play: {self_play_cmd}')
            subprocess_util.run(self_play_cmd, stdout=log_file,
                                stderr=log_file, check=True)
            logger.info(f'Gen-0 self-play complete!')

    def start(self, msg):
        assert self.child_process is None

        # TODO: once we change c++ to directly communicate game data to cmd-server via TCP, we will
        # no longer need games_base_dir or model here
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
            '--cmd-server-host', self.cmd_server_host,
            '--cmd-server-port', self.cmd_server_port,
            '--starting-generation', gen,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play-{self.client_id}.log')
        log_file = open(log_filename, 'a')
        proc = subprocess_util.Popen(self_play_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        self.child_process = proc
        logger.info(f'Running self-play [{proc.pid}]: {self_play_cmd}')

    def quit(self):
        logger.info(f'Received quit command')
        self._shutdown_code = 0

    def shutdown(self, code):
        logger.info(f'Shutting down...')
        if self.cmd_server_socket:
            self.cmd_server_socket.close()
        sys.exit(code)
