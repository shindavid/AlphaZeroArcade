"""
The DirectoryOrganizer class provides structured access to the contents of an alphzero directory.
Below is a diagram of the directory structure.

BASE_DIR/  # /workspace/output/game/tag/
    databases/
        clients.db
        ratings.db
        self-play.db
        training.db
    gens/
        gen-1/
            checkpoint.pt
            model.pt
            self-play/
                client-1/
                    {timestamp}.log
                    ...
                client-2/
                    ...
        gen-2/
            ...
        ...
    logs/
        loop-controller.log
        self-play-server/
            ...
        self-play-worker/
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
    def __init__(self, args: RunParams):
        """
        This constructor should not actually do any filesystem reading or writing. It should just
        set data members corresonding to expected filesystem paths.
        """
        game = args.game
        tag = args.tag

        self.game = game
        self.tag = tag

        self.base_dir = os.path.join('/workspace/output', game, tag)
        self.databases_dir = os.path.join(self.base_dir, 'databases')
        self.gens_dir = os.path.join(self.base_dir, 'gens')
        self.logs_dir = os.path.join(self.base_dir, 'logs')

        self.clients_db_filename = os.path.join(self.databases_dir, 'clients.db')
        self.ratings_db_filename = os.path.join(self.databases_dir, 'ratings.db')
        self.self_play_db_filename = os.path.join(self.databases_dir, 'self-play.db')
        self.training_db_filename = os.path.join(self.databases_dir, 'training.db')

        self.fork_info_filename = os.path.join(self.base_dir, 'fork-info.json')
        self._fork_info = None
        self._fork_info_loaded = False

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

    def requires_retraining(self):
        return self.fork_info is not None and len(self.fork_info.train_windows) > 0

    def makedirs(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.databases_dir, exist_ok=True)
        os.makedirs(self.gens_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _get_gens_dir(self, gen: Generation) -> str:
        return os.path.join(self.gens_dir, f'gen-{gen}')

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self._get_gens_dir(gen), 'model.pt')

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self._get_gens_dir(gen), 'checkpoint.pt')

    @staticmethod
    def get_ordered_subpaths(path: str) -> List[str]:
        subpaths = list(natsorted(f for f in os.listdir(path)))
        return [f for f in subpaths if not f.startswith('.')]

    @staticmethod
    def get_latest_full_subpath(path: str) -> Optional[str]:
        subpaths = DirectoryOrganizer.get_ordered_subpaths(path)
        return os.path.join(path, subpaths[-1]) if subpaths else None

    @staticmethod
    def get_latest_info(path: str) -> Optional[PathInfo]:
        subpaths = DirectoryOrganizer.get_ordered_subpaths(path)
        if not subpaths:
            return None
        return PathInfo(subpaths[-1])

    def _get_latest_generation(self, filename: str, default=None) -> Optional[Generation]:
        subpaths = DirectoryOrganizer.get_ordered_subpaths(self.gens_dir)
        for subpath in reversed(subpaths[-2:]):
            assert subpath.startswith('gen-'), f'Unexpected subpath: {subpath}'
            full_path = os.path.join(self.gens_dir, subpath, filename)
            if os.path.isfile(full_path):
                gen = int(subpath.split('-')[1])
                return gen
        return default

    def get_last_checkpointed_generation(self) -> Optional[Generation]:
        return self._get_latest_generation('checkpoint.pt')

    def get_latest_model_generation(self) -> Generation:
        return self._get_latest_generation('model.pt', default=0)

    def get_self_play_data_dir(self, client_id: ClientId, gen: Generation) -> str:
        return os.path.join(self._get_gens_dir(gen), f'self-play/client-{client_id}')

    def get_any_self_play_data_filename(self, gen: Optional[Generation]) -> Optional[str]:
        """
        Returns a self-play data filename for the given generation.

        If gen is None, returns a filename for the most recent generation that has self-play data.

        If no self-play data filename exists, returns None
        """
        if gen is None:
            gen = self.get_latest_model_generation()

        self_play_dir = os.path.join(self._get_gens_dir(gen), 'self-play')
        for client_subdir in os.listdir(self_play_dir):
            if not client_subdir.startswith('client-'):
                continue
            client_dir = os.path.join(self_play_dir, client_subdir)
            if os.path.isdir(client_dir):
                data_files = os.listdir(client_dir)
                if data_files:
                    return os.path.join(client_dir, data_files[0])

        if gen == 0:
            return None
        return self.get_any_self_play_data_filename(gen - 1)

    def copy_self_play_data(self, target: 'DirectoryOrganizer', last_gen: Optional[Generation]):
        for gen_subdir in os.listdir(self.gens_dir):
            assert gen_subdir.startswith('gen-'), f'Unexpected subpath: {gen_subdir}'
            gen = int(gen_subdir.split('-')[1])
            if gen > last_gen:
                continue
            src_gen_dir = os.path.join(self.gens_dir, gen_subdir)
            src_self_play_dir = os.path.join(src_gen_dir, 'self-play')

            dst_gen_dir = os.path.join(target.gens_dir, gen_subdir)
            dst_self_play_dir = os.path.join(dst_gen_dir, 'self-play')
            os.makedirs(dst_self_play_dir, exist_ok=True)

            shutil.copytree(src_self_play_dir, dst_self_play_dir)

    def copy_models_and_checkpoints(self, target: 'DirectoryOrganizer',
                                    last_gen: Optional[Generation]):
        if last_gen is None:
            last_gen = self.get_latest_model_generation()

        for gen in range(1, last_gen + 1):
            model_filename = self.get_model_filename(gen)
            target_model_filename = target.get_model_filename(gen)
            os.makedirs(os.path.dirname(target_model_filename), exist_ok=True)
            shutil.copyfile(model_filename, target_model_filename)

            checkpoint_filename = self.get_checkpoint_filename(gen)
            target_checkpoint_filename = target.get_checkpoint_filename(gen)
            shutil.copyfile(checkpoint_filename, target_checkpoint_filename)

    def copy_databases(self, target: 'DirectoryOrganizer', retrain_models: bool,
                       last_gen: Optional[Generation]):
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
