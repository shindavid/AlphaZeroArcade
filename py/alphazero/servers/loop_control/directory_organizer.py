"""
The DirectoryOrganizer class provides structured access to the contents of an alphazero directory.
Below is a diagram of the directory structure.

BASE_DIR/  # $OUTPUT_DIR/game/tag/
    version_file
    checkpoints/
        gen-1.pt
        gen-2.pt
        ...
    databases/
        clients.db
        ratings.db
        self-play.db
        training.db
    logs/
        loop-controller.log
        self-play-server/
            ...
        self-play-worker/
            ...
        ...
    models/
        gen-1.pt
        gen-2.pt
        ...
    self-play-data/
        gen-0/  # uses implicit dummy uniform model
            client-4/
                {timestamp}.log
                ...
        gen-1/
            client-5/
                ...
            ...
        ...
"""
from alphazero.logic.custom_types import ClientId, Generation
from alphazero.logic.run_params import RunParams
from util import sqlite3_util

from natsort import natsorted

import json
import os
import shutil
import sqlite3
from typing import Dict, List, Optional, Tuple


# VERSION is stored in the version_file in the base directory
#
# Any time we make any changes that cause existing output/ directories to be incompatible with the
# current code, we should increment VERSION.
#
# This should be a last resort that we try to avoid, but it's here in case we need it.
VERSION = 1


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class ForkInfo:
    def __init__(self, forked_base_dir: str):
        self.forked_base_dir: str = forked_base_dir
        self.max_client_id: int = -1
        self.train_windows: Dict[Generation, Tuple[int, int]] = {}

    def save(self, filename: str):
        json_dict = {
            'forked_base_dir': self.forked_base_dir,
            'max_client_id': self.max_client_id,
            'train_windows': self.train_windows,
        }

        with open(filename, 'w') as f:
            json.dump(json_dict, f, indent=4)

    @staticmethod
    def load(filename: str) -> 'ForkInfo':
        with open(filename, 'r') as f:
            json_dict = json.load(f)

        fork_info = ForkInfo(json_dict['forked_base_dir'])
        fork_info.max_client_id = json_dict['max_client_id']
        fork_info.train_windows = {int(k):v for k, v in json_dict['train_windows'].items()}
        return fork_info


class DirectoryOrganizer:
    def __init__(self, args: RunParams, base_dir_root='/home/devuser/scratch'):
        """
        This constructor should not actually do any filesystem reading or writing. It should just
        set data members corresonding to expected filesystem paths.
        """
        game = args.game
        tag = args.tag

        self.args = args

        self.base_dir_root = base_dir_root
        self.base_dir = os.path.join(base_dir_root, 'output', game, tag)
        self.databases_dir = os.path.join(self.base_dir, 'databases')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')
        self.misc_dir = os.path.join(self.base_dir, 'misc')

        self.clients_db_filename = os.path.join(self.databases_dir, 'clients.db')
        self.ratings_db_filename = os.path.join(self.databases_dir, 'ratings.db')
        self.self_play_db_filename = os.path.join(self.databases_dir, 'self-play.db')
        self.training_db_filename = os.path.join(self.databases_dir, 'training.db')

        self.version_filename = os.path.join(self.misc_dir, 'version_file')

        self.fork_info_filename = os.path.join(self.misc_dir, 'fork-info.json')
        self._fork_info = None
        self._fork_info_loaded = False

    @property
    def game(self) -> str:
        return self.args.game

    @property
    def tag(self) -> str:
        return self.args.tag

    @property
    def fork_info(self) -> ForkInfo:
        if not self._fork_info_loaded:
            if os.path.isfile(self.fork_info_filename):
                self._fork_info = ForkInfo.load(self.fork_info_filename)
            self._fork_info_loaded = True
        return self._fork_info

    @fork_info.setter
    def fork_info(self, value: ForkInfo):
        self._fork_info = value
        self._fork_info_loaded = True

    def version_check(self):
        """
        Checks that the version file matches VERSION. A nonexistent misc/ directory indicates that
        the dir hasn't been setup yet, which is a passing check. But if the misc/ directory exists
        while the version file doesn't, that is a failing check.

        Returns True if the check passes, and False if it fails.
        """
        if not os.path.exists(self.misc_dir):
            return True

        if os.path.isfile(self.version_filename):
            try:
                with open(self.version_filename, 'r') as f:
                    version = int(f.read())
                    if version == VERSION:
                        return True
            except:
                pass

        return False

    def requires_retraining(self):
        return self.fork_info is not None and len(self.fork_info.train_windows) > 0

    def dir_setup(self):
        """
        Performs initial setup of the directory structure.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.databases_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.misc_dir, exist_ok=True)

        if not os.path.isfile(self.version_filename):
            with open(self.version_filename, 'w') as f:
                f.write(str(VERSION))

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.models_dir, f'gen-{gen}.pt')

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self.checkpoints_dir, f'gen-{gen}.pt')

    @staticmethod
    def get_ordered_subpaths(path: str) -> List[str]:
        subpaths = list(natsorted(f for f in os.listdir(path)))
        return [f for f in subpaths if not f.startswith('.')]

    @staticmethod
    def get_latest_full_subpath(path: str) -> Optional[str]:
        subpaths = DirectoryOrganizer.get_ordered_subpaths(path)
        return os.path.join(path, subpaths[-1]) if subpaths else None

    @staticmethod
    def _get_latest_generation(path: str, default=None) -> Optional[Generation]:
        subpaths = DirectoryOrganizer.get_ordered_subpaths(path)
        if not subpaths:
            return default
        return PathInfo(subpaths[-1]).generation

    def get_last_checkpointed_generation(self, default=None) -> Optional[Generation]:
        return DirectoryOrganizer._get_latest_generation(self.checkpoints_dir, default=default)

    def get_latest_model_generation(self, default=None) -> Optional[Generation]:
        return DirectoryOrganizer._get_latest_generation(self.checkpoints_dir, default=default)

    def get_latest_self_play_generation(self, default=None) -> Optional[Generation]:
        return DirectoryOrganizer._get_latest_generation(self.self_play_data_dir, default=default)

    def get_self_play_data_dir(self, gen: Generation, client_id: Optional[ClientId]=None) -> str:
        gen_dir = os.path.join(self.self_play_data_dir, f'gen-{gen}')
        if client_id is None:
            return gen_dir
        return os.path.join(gen_dir, f'client-{client_id}')

    def get_any_self_play_data_filename(self, gen: Optional[Generation]) -> Optional[str]:
        """
        Returns a self-play data filename for the given generation.

        If gen is None, returns a filename for the most recent generation that has self-play data.

        If no self-play data filename exists, returns None
        """
        if gen is None:
            gen = self.get_latest_model_generation(default=0)

        gen_dir = os.path.join(self.self_play_data_dir, f'gen-{gen}')
        for client_subdir in os.listdir(self.self_play_data_dir):
            if not client_subdir.startswith('client-'):
                continue
            client_dir = os.path.join(gen_dir, client_subdir)
            if os.path.isdir(client_dir):
                data_files = os.listdir(client_dir)
                if data_files:
                    return os.path.join(client_dir, data_files[0])

        if gen == 0:
            return None
        return self.get_any_self_play_data_filename(gen - 1)

    def copy_self_play_data(self, target: 'DirectoryOrganizer',
                            last_gen: Optional[Generation]=None):
        for gen_subdir in os.listdir(self.self_play_data_dir):
            assert gen_subdir.startswith('gen-'), f'Unexpected subpath: {gen_subdir}'
            gen = int(gen_subdir.split('-')[1])
            if gen > last_gen:
                continue
            src_gen_dir = os.path.join(self.self_play_data_dir, gen_subdir)
            src_self_play_dir = os.path.join(src_gen_dir, 'self-play')

            dst_gen_dir = os.path.join(target.self_play_data_dir, gen_subdir)
            dst_self_play_dir = os.path.join(dst_gen_dir, 'self-play')
            os.makedirs(dst_self_play_dir, exist_ok=True)

            shutil.copytree(src_self_play_dir, dst_self_play_dir)

    def copy_models_and_checkpoints(self, target: 'DirectoryOrganizer',
                                    last_gen: Optional[Generation]=None):
        if last_gen is None:
            last_gen = self.get_latest_model_generation(default=0)

        for gen in range(1, last_gen + 1):
            model_filename = self.get_model_filename(gen)
            target_model_filename = target.get_model_filename(gen)
            shutil.copyfile(model_filename, target_model_filename)

            checkpoint_filename = self.get_checkpoint_filename(gen)
            target_checkpoint_filename = target.get_checkpoint_filename(gen)
            shutil.copyfile(checkpoint_filename, target_checkpoint_filename)

    def copy_databases(self, target: 'DirectoryOrganizer', retrain_models: bool=False,
                       last_gen: Optional[Generation]=None):
        shutil.copyfile(self.clients_db_filename, target.clients_db_filename)

        if not retrain_models:
            if last_gen is None:
                shutil.copyfile(self.ratings_db_filename, target.ratings_db_filename)
                shutil.copyfile(self.training_db_filename, target.training_db_filename)
            else:
                sqlite3_util.copy_db(self.ratings_db_filename, target.ratings_db_filename,
                                    f'mcts_gen <= {last_gen}')
                sqlite3_util.copy_db(self.training_db_filename, target.training_db_filename,
                                    f'gen <= {last_gen}')

        if last_gen is None:
            shutil.copyfile(self.self_play_db_filename, target.self_play_db_filename)
        else:
            sqlite3_util.copy_db(self.self_play_db_filename, target.self_play_db_filename,
                                 f'gen < {last_gen}')  # NOTE: intentionally using <, not <=

    def write_fork_info(self, from_organizer: 'DirectoryOrganizer', hard_fork: bool,
                        retrain_models: bool, last_gen: Optional[Generation]):
        self.fork_info = ForkInfo(from_organizer.base_dir)

        if not hard_fork:
            conn = sqlite3.connect(from_organizer.clients_db_filename)
            c = conn.cursor()
            c.execute('SELECT MAX(id) FROM clients')
            row = c.fetchone()
            if row:
                self.fork_info.max_client_id = row[0]
            conn.close()

        if retrain_models:
            conn = sqlite3.connect(from_organizer.training_db_filename)
            c = conn.cursor()
            if last_gen is not None:
                c.execute(f'SELECT gen, window_start, window_end FROM training '
                          f'WHERE gen <= {last_gen}')
            else:
                c.execute('SELECT gen, window_start, window_end FROM training')
            rows = c.fetchall()
            conn.close()

            for row in rows:
                gen, window_start, window_end = row
                self.fork_info.train_windows[gen] = (window_start, window_end)

        self.fork_info.save(self.fork_info_filename)
