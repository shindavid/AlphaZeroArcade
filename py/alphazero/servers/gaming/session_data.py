from .base_params import BaseParams

from alphazero.logic import constants
from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientId, ClientRole
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.socket_util import JsonDict, Socket
from util import py_util, ssh_util

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

        asset_requirements = data['asset-requirements']
        self._setup_run_directory(asset_requirements)

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

    def _setup_run_directory(self, asset_requirements: JsonDict):
        """
        asset_requirements takes the folowing form:

        {
            'binary': { path: hash },
            'extras': {
                path: hash,
                path: hash,
                ...
            }
        }

        This method does the following:

        1. Create an assets-directory in /home/devuser/scratch/assets/, if it doesn't yet exist.
           Files in this directory will be named by their hash.

        2. For each asset:

           A. Check if the assets-directory currently contains it. If so, continue.
           B. Else, check if the target/ directory contains it. If so, copy to the assets-directory,
              otherwise, continue
           C. If the asset is still missing, request it from the loop controller, and write the
              asset to the assets-directory

        3. Create a unique run-directory in /home/devuser/scratch/runs/ to be used by this server
           and this server only. Add symlinks in this run-directory, named by the asset path, that
           point to the asset in the assets-directory.

        4. Set appropriate data members to self, that support methods that SelfPlayServer and
           RatingsServer can use to construct their run-command-strings.
        """
        assets_dir = '/home/devuser/scratch/assets'
        py_util.atomic_makedirs(assets_dir)

        binary_info = asset_requirements['binary']
        extras_info = asset_requirements['extras']

        if self._build_params.binary_path is not None:
            self._binary_path = './custom-binary'

            src = os.path.abspath(self._build_params.binary_path)
            dst = os.path.join(self.run_dir, self._binary_path)
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            ln_cmd = f'ln -sf {src} {dst}'
            subprocess.run(ln_cmd, shell=True, check=True)
        else:
            assert len(binary_info) == 1
            self._binary_path = list(binary_info.keys())[0]

        hash_dict = {}
        missing_assets = []
        for info in [binary_info, extras_info]:
            hash_dict.update(info)
            for asset_path, asset_hash in info.items():
                # See 2A above
                dst_path = os.path.join(assets_dir, asset_hash)
                if os.path.exists(dst_path):
                    continue

                # See 2B above
                workspace_path = os.path.join('/workspace/repo', asset_path)
                if os.path.exists(workspace_path):
                    candidate_hash = py_util.sha256sum(workspace_path)
                    if candidate_hash == asset_hash:
                        py_util.atomic_cp(workspace_path, dst_path)
                        continue

                # See 2C above
                missing_assets.append(asset_path)

        data = {
            'type': 'assets-request',
            'assets': missing_assets,
        }
        self.socket.send_json(data)

        logger.info('DBG sent assets-request %s', data)
        for asset_path in missing_assets:
            asset_hash = hash_dict[asset_path]
            full_asset_path = os.path.join(assets_dir, asset_hash)
            self.socket.recv_file(full_asset_path, atomic=True)
            logger.info('DBG received asset: %s', full_asset_path)
            assert os.path.isfile(full_asset_path)

        # Create soft links to the assets. Note  that because the run_dir will be unique to each
        # server, we don't need to worry about atomicity for the filesytem operations below.
        for asset_path, asset_hash in hash_dict.items():
            src = os.path.join(assets_dir, asset_hash)
            dst = os.path.join(self.run_dir, asset_path)
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            ln_cmd = f'ln -sf {src} {dst}'
            subprocess.run(ln_cmd, shell=True, check=True)

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
