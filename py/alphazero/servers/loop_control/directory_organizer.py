"""
The DirectoryOrganizer class provides structured access to the contents of an alphazero directory.
Below is a diagram of the directory structure.

output/
├── {game}/
│   ├── {tag}/
│   │   ├── bin/{game} (game binary)
│   │   ├── checkpoints/
│   │   │   ├── gen-1.pt
│   │   │   ├── gen-2.pt
│   │   │   └── ...
│   │   ├── databases/
│   │   │   ├── evaluation/
│   │   │   │   ├── {benchmark_tag}.db
│   │   │   │   └── ...
│   │   │   ├── benchmark.db
│   │   │   ├── clients.db
│   │   │   ├── self-play.db
│   │   │   └── training.db
│   │   ├── logs/
│   │   │   ├── benchmark-server/
│   │   │   ├── benchmark-worker/
│   │   │   ├── eval-server/
│   │   │   ├── eval-worker/
│   │   │   ├── gen0-self-play-worker/
│   │   │   ├── self-play-server/
│   │   │   ├── self-play-worker/
│   │   │   ├── loop-controller.log
│   │   ├── misc/version_file
│   │   ├── models/
│   │   │   ├── gen-1.onnx
│   │   │   ├── gen-2.onnx
│   │   │   └── ...
│   │   ├── runtime/
│   │   │   ├── lock (optional, indicates a running process)
│   │   │   └── freeze (optional, indicates a frozen run)
│   │   └── self-play-data/
│   │       ├── gen-1.data
│   │       ├── gen-2.data
│   │       └── ...
│   └── ...
└── ...
"""
from alphazero.logic.custom_types import Generation
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import BaseDir

from natsort import natsorted

from dataclasses import asdict, dataclass
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
PathStr = str


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
        self.train_windows: Dict[Generation, Tuple[int, int]] = {}

    def save(self, filename: str):
        json_dict = {
            'forked_base_dir': self.forked_base_dir,
            'train_windows': self.train_windows,
        }

        with open(filename, 'w') as f:
            json.dump(json_dict, f, indent=4)

    @staticmethod
    def load(filename: str) -> 'ForkInfo':
        with open(filename, 'r') as f:
            json_dict = json.load(f)

        fork_info = ForkInfo(json_dict['forked_base_dir'])
        fork_info.train_windows = {int(k): v for k, v in json_dict['train_windows'].items()}
        return fork_info

@dataclass
class VersionInfo:
    paradigm: Optional[str] = None

    def write_to_file(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(filename: str) -> 'VersionInfo':
        try:
            with open(filename, 'r') as f:
                json_dict = json.load(f)
            return VersionInfo(paradigm=json_dict.get('paradigm'))
        except Exception as e:
            logger.error(f'Error loading version info from {filename}: {e}')
            return VersionInfo(paradigm=None)

class DirectoryOrganizer:
    def __init__(self, args: RunParams, base_dir_root: BaseDir):
        """
        This constructor should not actually do any filesystem reading or writing. It should just
        set data members corresonding to expected filesystem paths.
        """
        game = args.game
        tag = args.tag

        self.args = args

        self.base_dir_root = base_dir_root.output_dir()
        self.game_dir = os.path.join(self.base_dir_root, game)
        self.base_dir = os.path.join(self.game_dir, tag)
        self.databases_dir = os.path.join(self.base_dir, 'databases')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')
        self.misc_dir = os.path.join(self.base_dir, 'misc')
        self.eval_db_dir = os.path.join(self.databases_dir, 'evaluation')
        self.runtime_dir = os.path.join(self.base_dir, 'runtime')
        self.binary_dir = os.path.join(self.base_dir, 'bin')

        self.clients_db_filename = os.path.join(self.databases_dir, 'clients.db')
        self.ratings_db_filename = os.path.join(self.databases_dir, 'ratings.db')
        self.self_play_db_filename = os.path.join(self.databases_dir, 'self-play.db')
        self.training_db_filename = os.path.join(self.databases_dir, 'training.db')
        self.benchmark_db_filename = os.path.join(self.databases_dir, 'benchmark.db')

        self.binary_filename = os.path.join(self.binary_dir, game)
        self.version_filename = os.path.join(self.misc_dir, 'version_file')
        self.lock_filename = os.path.join(self.runtime_dir, 'lock')
        self.freeze_filename = os.path.join(self.runtime_dir, 'freeze')

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

    def eval_db_filename(self, benchmark_tag: str) -> str:
        return os.path.join(self.eval_db_dir, f'{benchmark_tag}.db')

    def requires_retraining(self):
        return self.fork_info is not None and len(self.fork_info.train_windows) > 0

    def dir_setup(self, benchmark_tag: Optional[str] = None):
        """
        Performs initial setup of the directory structure.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.databases_dir, exist_ok=True)
        os.makedirs(self.misc_dir, exist_ok=True)

        if benchmark_tag != 'reference.player':
            os.makedirs(self.binary_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)

        if benchmark_tag is None:
            os.makedirs(self.eval_db_dir, exist_ok=True)
            os.makedirs(self.self_play_data_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            os.makedirs(self.misc_dir, exist_ok=True)
            os.makedirs(self.runtime_dir, exist_ok=True)

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.models_dir, f'gen-{gen}.onnx')

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self.checkpoints_dir, f'gen-{gen}.pt')

    @staticmethod
    def find_latest_tag(game: str, base_dir_root: BaseDir,
                        paradigm: Optional[str] = None) -> Optional[str]:
        output_path = Path(base_dir_root.output_dir()) / game

        if not output_path.is_dir():
            return None

        valid_dirs: List[Path] = []
        for entry in output_path.iterdir():
            if not entry.is_dir():
                continue

            if paradigm:
                organizer = DirectoryOrganizer(
                    RunParams(game=game, tag=entry.name),
                    base_dir_root,
                )
                organizer_paradigm = organizer.paradigm()
                if organizer_paradigm != paradigm:
                    continue

            valid_dirs.append(entry)

        if not valid_dirs:
            return None

        latest_dir = max(valid_dirs, key=lambda d: d.stat().st_mtime)
        return latest_dir.name

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

    @staticmethod
    def get_gen_number(filepath: str) -> Generation:
        """
        Extracts the generation number from a filename.
        filepath is expected to be like {path}/gen-123.{ext}
        """
        stem = Path(filepath).stem
        left, right = stem.split('-', 1)
        if left != 'gen' or not right.isdigit():
            raise ValueError(f'Unexpected file: {filepath}')
        return int(right)

    def get_last_checkpointed_generation(self, default=None) -> Optional[Generation]:
        return DirectoryOrganizer._get_latest_generation(self.checkpoints_dir, default=default)

    def get_latest_model_generation(self, default=None) -> Optional[Generation]:
        return DirectoryOrganizer._get_latest_generation(self.checkpoints_dir, default=default)

    def get_latest_self_play_generation(self, default=None) -> Optional[Generation]:
        return DirectoryOrganizer._get_latest_generation(self.self_play_data_dir, default=default)

    def get_self_play_data_filename(self, gen: Generation) -> str:
        return os.path.join(self.self_play_data_dir, f'gen-{gen}.data')

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

    def _apply_to_self_play_data_dir(self, target: 'DirectoryOrganizer',
                                     func: Callable[[PathStr, PathStr], Any],
                                     last_model_gen: Optional[Generation] = None):
        for genfile in os.listdir(self.self_play_data_dir):
            gen = DirectoryOrganizer.get_gen_number(genfile)

            if last_model_gen is not None and gen >= last_model_gen:
                continue
            src = os.path.join(self.self_play_data_dir, genfile)
            dst = os.path.join(target.self_play_data_dir, genfile)
            func(src, dst)

    def copy_self_play_data(self, target: 'DirectoryOrganizer',
                            last_model_gen: Optional[Generation] = None):
        self._apply_to_self_play_data_dir(target, shutil.copyfile, last_model_gen)

    def soft_link_self_play_data(self, target: 'DirectoryOrganizer',
                                 last_model_gen: Optional[Generation] = None):
        self._apply_to_self_play_data_dir(target, os.symlink, last_model_gen)

    def copy_models_and_checkpoints(self, target: 'DirectoryOrganizer',
                                    last_gen: Optional[Generation] = None):
        if last_gen is None:
            last_gen = self.get_latest_model_generation(default=0)

        for gen in range(1, last_gen + 1):
            model_filename = self.get_model_filename(gen)
            target_model_filename = target.get_model_filename(gen)
            shutil.copyfile(model_filename, target_model_filename)

            checkpoint_filename = self.get_checkpoint_filename(gen)
            target_checkpoint_filename = target.get_checkpoint_filename(gen)
            shutil.copyfile(checkpoint_filename, target_checkpoint_filename)

    def soft_link_models_and_checkpoints(self, target: 'DirectoryOrganizer',
                                         last_gen: Optional[Generation] = None):
        if last_gen is None:
            last_gen = self.get_latest_model_generation(default=0)

        for gen in range(1, last_gen + 1):
            model_filename = self.get_model_filename(gen)
            target_model_filename = target.get_model_filename(gen)
            os.symlink(model_filename, target_model_filename)

            checkpoint_filename = self.get_checkpoint_filename(gen)
            target_checkpoint_filename = target.get_checkpoint_filename(gen)
            os.symlink(checkpoint_filename, target_checkpoint_filename)

    def copy_binary(self, target: 'DirectoryOrganizer'):
        if not os.path.isfile(self.binary_filename):
            raise ValueError(f'Binary file does not exist: {self.binary_filename}')
        shutil.copyfile(self.binary_filename, target.binary_filename)

    def write_fork_info(self, from_organizer: 'DirectoryOrganizer',
                        retrain_models: bool, last_gen: Optional[Generation]):
        self.fork_info = ForkInfo(from_organizer.base_dir)

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

    def acquire_lock(self, register_func: Callable) -> str:
        self.assert_unlocked()

        with open(self.lock_filename, 'w') as f:
            f.write('The existence of this file indicates that this run is currently active.')
        logger.debug(f"Lock acquired: {self.lock_filename}")
        register_func(self.release_lock, 'lock-release')

    def release_lock(self):
        if os.path.exists(self.lock_filename):
            os.remove(self.lock_filename)
            logger.info(f"Lock {self.lock_filename} released.")

    def assert_not_frozen(self):
        if os.path.exists(self.freeze_filename):
            raise RuntimeError(
                f"game {self.game} tag {self.tag} is frozen.\n"
                f"To unfreeze, remove the freeze file in "
                f"{self.freeze_filename}")

    def assert_unlocked(self):
        if os.path.exists(self.lock_filename):
            raise RuntimeError(
                f"game {self.game} tag {self.tag} is locked.\n"
                f"To unlock, remove the lock file in "
                f"{self.lock_filename}")

    def freeze_tag(self):
        with open(self.freeze_filename, 'w') as f:
            f.write('The existence of this file indicates that this run was benchmarked, and thus \
                    that no more models can be trained for this tag.')
        logger.info(f"Froze run {self.game}: {self.tag}.")

    def write_version_file(self, paradigm: str):
        if Path(self.version_filename).exists():
            return
        version_info = VersionInfo(paradigm=paradigm)
        version_info.write_to_file(self.version_filename)

    def paradigm(self) -> Optional[str]:
        if not Path(self.version_filename).exists():
            return None
        version_info = VersionInfo.load(self.version_filename)
        return version_info.paradigm
