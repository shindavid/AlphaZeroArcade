"""
The DirectoryOrganizer class provides structured access to the contents of an alphzero directory.
Below is a diagram of the directory structure.

BASE_DIR/  # $alphazero_dir/game/tag/
    stdout.txt
    databases/
        ratings.db
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
    bins/
        {hash1}
        {hash2}
        ...
    checkpoints/
        gen-1.pt
        gen-2.pt
        ...
"""
from alphazero.custom_types import Generation
from alphazero.common_args import CommonArgs

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
    def __init__(self):
        alphazero_dir = CommonArgs.alphazero_dir
        game = CommonArgs.game
        tag = CommonArgs.tag

        self.base_dir = os.path.join(alphazero_dir, game, tag)
        self.databases_dir = os.path.join(self.base_dir, 'databases')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.bins_dir = os.path.join(self.base_dir, 'bins')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')

        self.ratings_db_filename = os.path.join(self.databases_dir, 'ratings.db')
        self.training_db_filename = os.path.join(self.databases_dir, 'training.db')

    def makedirs(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.databases_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.bins_dir, exist_ok=True)
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

    def get_latest_binary(self):
        bins = [os.path.join(self.bins_dir, b)
                for b in os.listdir(self.bins_dir)]
        bins.sort(key=os.path.getmtime)
        return bins[-1]
