"""
The DirectoryOrganizer class provides structured access to the contents of an alphzero directory.
Below is a diagram of the directory structure.

BASE_DIR/  # $A0A_OUTPUT_DIR/game/tag/
    stdout.txt
    databases/
        clients.db
        ratings.db
        self-play.db
        training.db
    self-play-data/
        client-0/
            gen-0/  # uses implicit dummy uniform model
                {timestamp}.log
                ...
            gen-1/  # uses models/gen-1.pt
                ...
            gen-2/  # uses models/gen-2.pt
                ...
            ...
        client-1/
            gen-3/
                ...
            ...
        ...
    models/
        gen-1.pt
        gen-2.pt
        ...
    checkpoints/
        gen-1.pt
        gen-2.pt
        ...
"""
from alphazero.logic.custom_types import Generation
from alphazero.logic.run_params import RunParams
from util.env_util import get_output_dir
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
        game = args.game
        tag = args.tag

        self.game = game
        self.tag = tag

        self.base_dir = os.path.join(get_output_dir(), game, tag)
        self.databases_dir = os.path.join(self.base_dir, 'databases')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')

        self.clients_db_filename = os.path.join(self.databases_dir, 'clients.db')
        self.ratings_db_filename = os.path.join(self.databases_dir, 'ratings.db')
        self.self_play_db_filename = os.path.join(self.databases_dir, 'self-play.db')
        self.training_db_filename = os.path.join(self.databases_dir, 'training.db')

        self.fork_info_filename = os.path.join(self.base_dir, 'fork-info.json')
        self.fork_info = None
        if os.path.isfile(self.fork_info_filename):
            self.fork_info = ForkInfo.load(self.fork_info_filename)

    def requires_retraining(self):
        return self.fork_info is not None and len(self.fork_info.train_windows) > 0

    def makedirs(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.databases_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

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
    def get_latest_info(path: str) -> Optional[PathInfo]:
        subpaths = DirectoryOrganizer.get_ordered_subpaths(path)
        if not subpaths:
            return None
        return PathInfo(subpaths[-1])

    def get_latest_model_info(self) -> Optional[PathInfo]:
        return DirectoryOrganizer.get_latest_info(self.models_dir)

    def get_latest_checkpoint_info(self) -> Optional[PathInfo]:
        return DirectoryOrganizer.get_latest_info(self.checkpoints_dir)

    def get_latest_model_generation(self) -> Generation:
        info = DirectoryOrganizer.get_latest_info(self.models_dir)
        return 0 if info is None else info.generation

    def get_latest_model_filename(self) -> Optional[str]:
        return DirectoryOrganizer.get_latest_full_subpath(self.models_dir)

    def copy_self_play_data(self, target: 'DirectoryOrganizer', last_gen: Optional[Generation]):
        for client_subdir in os.listdir(self.self_play_data_dir):
            if not client_subdir.startswith('client-'):
                continue
            client_dir = os.path.join(self.self_play_data_dir, client_subdir)
            target_client_dir = os.path.join(target.self_play_data_dir, client_subdir)
            os.makedirs(target_client_dir, exist_ok=True)

            for gen_subdir in os.listdir(client_dir):
                if not gen_subdir.startswith('gen-'):
                    continue
                gen_dir = os.path.join(client_dir, gen_subdir)
                gen = int(gen_subdir.split('-')[1])
                if last_gen is not None and gen > last_gen:
                    continue

                target_gen_dir = os.path.join(target_client_dir, gen_subdir)
                shutil.copytree(gen_dir, target_gen_dir)

    def copy_models_and_checkpoints(self, target: 'DirectoryOrganizer',
                                    last_gen: Optional[Generation]):
        if last_gen is None:
            last_gen = self.get_latest_model_generation()

        for gen in range(1, last_gen + 1):
            model_filename = self.get_model_filename(gen)
            target_model_filename = target.get_model_filename(gen)
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
