from .base_params import BaseParams

from alphazero.logic import constants
from alphazero.logic.custom_types import ClientId, ClientRole
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.socket_util import Socket
from util import ssh_util
from util import subprocess_util

import os
import socket
import subprocess
import time
from typing import Optional


logger = get_logger()


class SessionData:
    """
    Connecting with the loop controller leads to the creation of a session.

    This class holds various data that is associated with that session.
    """
    def __init__(self, params: BaseParams, logging_params: LoggingParams):
        self._params = params
        self._logging_params = logging_params
        self._game = None
        self._game_spec = None
        self._tag = None
        self._directory_organizer = None
        self._socket: Optional[Socket] = None
        self._client_id: Optional[ClientId] = None
        self._skip_next_returncode_check = False
        self._log_sync_set = set()

    def disable_next_returncode_check(self):
        self._skip_next_returncode_check = True

    def wait_for(self, proc: subprocess.Popen):
        stdout, stderr = proc.communicate()
        if self._skip_next_returncode_check:
            self._skip_next_returncode_check = False
        elif proc.returncode:
            logger.error(f'Process failed with return code {proc.returncode}')
            for line in stderr:
                logger.error(line.strip())
            raise Exception()
        return stdout

    def init_socket(self):
        addr = (self._params.loop_controller_host, self._params.loop_controller_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # run_local.py launches the servers together. The loop-controller socket might not be
        # fully initialized by the time the other servers reach this point. So we retry a few times
        # to connect to the loop-controller.
        retry_count = 5
        sleep_time = 1
        connected = False
        for _ in range(retry_count):
            try:
                sock.connect(addr)
                connected = True
                break
            except:
                time.sleep(sleep_time)
        if not connected:
            raise Exception(f'Failed to connect to: {addr}')

        self._socket = Socket(sock)

    def send_handshake(self, role: ClientRole, aux: Optional[dict] = None):
        data = {
            'type': 'handshake',
            'role': role.value,
            'start_timestamp': time.time_ns(),
            'cuda_device': self._params.cuda_device,
        }
        if aux is not None:
            data['aux'] = aux

        self.socket.send_json(data)

    def recv_handshake(self, role: ClientRole):
        data = self.socket.recv_json(timeout=1)
        assert data['type'] == 'handshake-ack', data

        rejection = data.get('rejection', None)
        if rejection is not None:
            raise Exception(f'Handshake rejected: {rejection}')

        self._game = data['game']
        self._tag = data['tag']
        self._client_id = data['client_id']

        if self.socket.getsockname()[0] == constants.LOCALHOST_IP:
            on_ephemeral_local_disk_env = data['on_ephemeral_local_disk_env']
            run_params = RunParams(self._game, self._tag)
            if on_ephemeral_local_disk_env:
                self._directory_organizer = DirectoryOrganizer(run_params)
            else:
                self._directory_organizer = DirectoryOrganizer(run_params,
                                                               base_dir_root='/workspace')

        ssh_pub_key = data['ssh_pub_key']
        ssh_util.add_to_authorized_keys(ssh_pub_key)

        log_filename = self.get_log_filename(role.value)
        configure_logger(params=self._logging_params, filename=log_filename, mode='w')
        self.start_log_sync(log_filename)
        logger.info('**** Starting %s ****', role.value)
        logger.info('Received client id assignment: %s', self._client_id)

    def get_log_filename(self, src: str):
        return os.path.join('/home/devuser/scratch/logs', self.game, self.tag, src,
                            f'{src}-{self.client_id}.log')

    def start_log_sync(self, log_filename):
        """
        Starts syncing the log file with the loop controller.

        Returns False if the log file is already being synced, and True otherwise
        """
        if log_filename in self._log_sync_set:
            return False

        self._log_sync_set.add(log_filename)
        data = {
            'type': 'log-sync-start',
            'log_filename': log_filename,
        }
        self.socket.send_json(data)
        return True

    def stop_log_sync(self, log_filename):
        if log_filename in self._log_sync_set:
            self._log_sync_set.remove(log_filename)

        data = {
            'type': 'log-sync-stop',
            'log_filename': log_filename,
        }
        self.socket.send_json(data)

    @property
    def socket(self) -> Socket:
        if self._socket is None:
            raise ValueError('loop controller socket not initialized')
        return self._socket

    @property
    def client_id(self) -> ClientId:
        if self._client_id is None:
            raise ValueError('client id not set')
        return self._client_id

    @property
    def game(self) -> str:
        if self._game is None:
            raise ValueError('game not set')
        return self._game

    @property
    def tag(self) -> str:
        if self._tag is None:
            raise ValueError('tag not set')
        return self._tag

    @property
    def game_spec(self) -> GameSpec:
        if self._game_spec is None:
            self._game_spec = get_game_spec(self.game)
        return self._game_spec

    @property
    def directory_organizer(self) -> Optional[DirectoryOrganizer]:
        # Return None if the loop-controller is not running on localhost
        return self._directory_organizer
