"""
The DirectoryOrganizer class provides structured access to the contents of an alphzero directory.
Below is a diagram of the directory structure.

BASE_DIR/  # $output_dir/game/tag/
    stdout.txt
    databases/
        clients.db
        ratings.db
        self-play.db
        training.db
    self-play-data/
        client-0/
            gen-0/  # uses implicit dummy uniform model
                {timestamp}.pt
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

import os
from typing import List, Optional

from natsort import natsorted


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class DirectoryOrganizer:
    def __init__(self, args: RunParams):
        output_dir = args.output_dir
        game = args.game
        tag = args.tag

        self.base_dir = os.path.join(output_dir, game, tag)
        self.databases_dir = os.path.join(self.base_dir, 'databases')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')

        self.clients_db_filename = os.path.join(self.databases_dir, 'clients.db')
        self.ratings_db_filename = os.path.join(self.databases_dir, 'ratings.db')
        self.self_play_db_filename = os.path.join(self.databases_dir, 'self-play.db')
        self.training_db_filename = os.path.join(self.databases_dir, 'training.db')

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

    def get_latest_generation(self) -> Generation:
        return self.get_latest_model_generation()

    def get_latest_model_filename(self) -> Optional[str]:
        return DirectoryOrganizer.get_latest_full_subpath(self.models_dir)
