from .base_params import BaseParams

from alphazero.logic import constants
from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientId, ClientRole, FileToTransfer
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import LoggingParams, configure_logger
from util.socket_util import JsonDict, Socket
from util import py_util, ssh_util

import logging
import os
import socket
import subprocess
import threading
import time
from typing import Optional, Dict, List


logger = logging.getLogger(__name__)


ASSETS_DIR = '/home/devuser/scratch/assets'


class SessionData:
    """
    Connecting with the loop controller leads to the creation of a session.

    This class holds various data that is associated with that session.
    """
    def __init__(self, params: BaseParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        self._params = params
        self._logging_params = logging_params
        self._build_params = build_params
        self._game = None
        self._game_spec = None
        self._tag = None
        self._binary_path = None
        self._directory_organizer = None
        self._socket: Optional[Socket] = None
        self._client_id: Optional[ClientId] = None
        self._skip_next_returncode_check = False
        self._log_sync_set = set()
        self._file_transfer_cv = threading.Condition()

    def disable_next_returncode_check(self):
        self._skip_next_returncode_check = True

    def wait_for(self, proc: subprocess.Popen):
        stdout, stderr = proc.communicate()
        if self._skip_next_returncode_check:
            self._skip_next_returncode_check = False
        elif proc.returncode:
            logger.error(f'Process failed with return code {proc.returncode}')
            for line in stderr.splitlines():
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

    def send_handshake(self, role: ClientRole, rating_tag: str = ''):
        data = {
            'type': 'handshake',
            'role': role.value,
            'start_timestamp': time.time_ns(),
            'cuda_device': self._params.cuda_device,
        }
        if rating_tag:
            data['rating_tag'] = rating_tag

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

    def get_missing_binaries(self, binaries: List[JsonDict]):
        missing_binaries = []
        for b in binaries:
            logger.debug('Checking binary: %s', b)
            binary = FileToTransfer(**b)
            asset_path = os.path.join(ASSETS_DIR, binary.hash)
            if not os.path.exists(asset_path):
                missing_binaries.append(binary)
            else:
                # Verify the hash matches if the file exists
                current_hash = py_util.sha256sum(asset_path)
                if current_hash != binary.hash:
                    logger.debug(f'Hash mismatch for binary {binary.source_path}: '
                                   f'expected {binary.hash}, found {current_hash}')
                    missing_binaries.append(binary)
                else:
                    # Create a soft link to the correct binary if necessary
                    dst_path = os.path.join(self.run_dir, binary.scratch_path)
                    dst_dir = os.path.dirname(dst_path)
                    os.makedirs(dst_dir, exist_ok=True)
                    ln_cmd = f'ln -sf {asset_path} {dst_path}'
                    subprocess.run(ln_cmd, shell=True, check=True)

        return missing_binaries

    def send_binary_request(self, missing_binaries: List[FileToTransfer]):
        if not missing_binaries:
            return

        binaries_info = [b.to_dict() for b in missing_binaries]
        data = {
            'type': 'binary-request',
            'binaries': binaries_info,
        }
        self.socket.send_json(data)
        logger.info('Sent binary-request for %d binaries: %s', len(missing_binaries), binaries_info)

    def receive_binary_file(self, binary_dict: JsonDict):
        py_util.atomic_makedirs(ASSETS_DIR)
        binary = FileToTransfer(**binary_dict)
        asset_path = os.path.join(ASSETS_DIR, binary.hash)
        self.socket.recv_file(asset_path, atomic=True)

        # create soft link in the run directory
        src = asset_path
        dst = os.path.join(self.run_dir, binary.scratch_path)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        ln_cmd = f'ln -sf {src} {dst}'
        subprocess.run(ln_cmd, shell=True, check=True)

        with self._file_transfer_cv:
            self._file_transfer_cv.notify_all()

    def wait_for_binaries(self, required_binaries: List[JsonDict]):
        with  self._file_transfer_cv:
            self._file_transfer_cv.wait_for(lambda: self.get_missing_binaries(required_binaries) == [])

    def is_model_missing(self, required_model: JsonDict):
        model_file = FileToTransfer(**required_model)
        if not model_file.scratch_path:
            return False

        asset_path = os.path.join(ASSETS_DIR, model_file.scratch_path)
        if not os.path.exists(asset_path):
            return True
        else:
            current_hash = py_util.sha256sum(asset_path)
            if current_hash != model_file.hash:
                logger.debug(f'Hash mismatch for model {model_file.source_path}: '
                             f'expected {model_file.hash}, found {current_hash}')
                return True
            else:
                dst_path = os.path.join(self.run_dir, model_file.scratch_path)
                dst_dir = os.path.dirname(dst_path)
                os.makedirs(dst_dir, exist_ok=True)
                ln_cmd = f'ln -sf {asset_path} {dst_path}'
                subprocess.run(ln_cmd, shell=True, check=True)

        return False

    def send_model_request(self, required_model: JsonDict):
        if not required_model:
            return

        data = {
            'type': 'model-request',
            'model': required_model,
        }
        self.socket.send_json(data)
        logger.info('Sent model-request: %s', data)

    def wait_for_model(self, required_model: JsonDict):
        with self._file_transfer_cv:
            self. _file_transfer_cv.wait_for(lambda: not self.is_model_missing(required_model))

    def receive_model_file(self, model_dict: JsonDict):
        py_util.atomic_makedirs(ASSETS_DIR)
        model_file = FileToTransfer(**model_dict)
        asset_path = os.path.join(ASSETS_DIR, model_file.scratch_path)
        os.makedirs(os.path.dirname(asset_path), exist_ok=True)
        self.socket.recv_file(asset_path, atomic=True)
        logger.info('Received model file: %s', asset_path)

        # create soft link in the run directory
        dst_path = os.path.join(self.run_dir, model_file.scratch_path)
        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)
        ln_cmd = f'ln -sf {asset_path} {dst_path}'
        subprocess.run(ln_cmd, shell=True, check=True)

        with self._file_transfer_cv:
            self._file_transfer_cv.notify_all()

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
    def binary_path(self) -> str:
        if self._binary_path is None:
            raise ValueError('binary path not set')
        return self._binary_path

    @property
    def run_dir(self) -> str:
        return os.path.join('/home/devuser/scratch/runs', self.game, self.tag, str(self.client_id))

    @property
    def directory_organizer(self) -> Optional[DirectoryOrganizer]:
        # Return None if the loop-controller is not running on localhost
        return self._directory_organizer
