from alphazero.logic.agent_type import IndexedAgent
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Workspace
from games.game_spec import GameSpec
from games.index import get_game_spec

from dataclasses import dataclass
import glob
import logger
import json
import os
import shlex
import shutil
import sys
from typing import Optional


UTC_FORMAT = '%Y-%m-%d_%H-%M-%S.%f_UTC'


@dataclass
class BenchmarkRecord:
    utc_key: Optional[str] = None
    tag: Optional[str] = None
    game: Optional[str] = None

    def to_dict(self):
        return {'utc_key': self.utc_key, 'tag': self.tag}

    def key(self):
        return os.path.join(self.game, self.tag, f"{self.utc_key}.tar")

    @staticmethod
    def load(game: str) -> Optional['BenchmarkRecord']:
        """
        This will read the file:
            /workspace/repo/benchmark_records/{game}.json
        """

        file_path = Workspace.benchmark_record_file(game)

        if not os.path.exists(file_path):
            logger.info(f"No benchmark record found for game '{game}' at {file_path}. ")
            return None

        with open(file_path, 'r') as f:
            benchmark_record = json.load(f)

        utc_key = benchmark_record.get("utc_key", None)
        tag = benchmark_record.get("tag", None)

        if utc_key is None or tag is None or hash is None:
            raise ValueError(f"Invalid benchmark info file format for game '{game}': {file_path}")

        return BenchmarkRecord(utc_key=utc_key, tag=tag, game=game)


class BenchmarkData:
    @staticmethod
    def path(game: str, tag: str, utc_key: str = None) -> Optional[str]:
        if utc_key is None:
            tag_dir = os.path.join(Workspace.benchmark_data_dir, game, tag)
            if not os.path.isdir(tag_dir):
                return None

            folders = os.listdir(tag_dir)
            if folders:
                utc_key = max(folders)
            else:
                return None
        return os.path.join(Workspace.benchmark_data_dir, game, tag, utc_key)

    @staticmethod
    def tar_path(game: str, tag: str, utc_key: str = None) -> Optional[str]:
        if utc_key is None:
            tag_dir = os.path.join(Workspace.benchmark_data_dir, game, tag)
            if not os.path.isdir(tag_dir):
                return None

            tar_files = glob.glob(os.path.join(tag_dir, '*.tar'))
            if tar_files:
                utc_key = max(tar_files)
            else:
                return None
        return os.path.join(Workspace.benchmark_data_dir, game, tag, utc_key)


class BenchmarkOption:
    def __init__(self, tag: str, game: str):
        self.tag = tag
        self.game = game

    def has_reference_player(self):
        game_spec: GameSpec = get_game_spec(self.game)
        return game_spec.reference_player_family is not None

    def on_record(self) -> Optional[BenchmarkRecord]:
        record: BenchmarkRecord = BenchmarkRecord.load(self.game)
        if record and record.tag == self.tag:
            return record
        return None

    def has_run_dir(self) -> bool:
        organizer = DirectoryOrganizer(RunParams(self.game, self.tag), base_dir_root=Workspace)
        return os.path.isdir(organizer.base_dir)

    @staticmethod
    def benchmark_folder(tag: str):
        return f'{tag}.benchmark'

    def has_benchmark_run_dir(self) -> bool:
        benchmark_folder = BenchmarkOption.benchmark_folder(self.tag)
        run_params = RunParams(self.game, benchmark_folder)
        organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
        return os.path.isdir(organizer.base_dir)

    def has_benchmark_data(self) -> bool:
        return BenchmarkData.path(self.game, self.tag) is not None

    def has_benchmark_tar_file(self) -> bool:
        return BenchmarkData.tar_path(self.game, self.tag) is not None


def save_benchmark_data(organizer: DirectoryOrganizer, record: BenchmarkRecord):
    benchmarker = Benchmarker(organizer)
    path = record.data_folder_path()
    model_path = os.path.join(path, 'models')
    os.makedirs(model_path, exist_ok=True)
    file = os.path.join(path, 'ratings.json')
    rating_data: BenchmarkRatingData = benchmarker.read_ratings_from_db()

    ix = 0
    db_id = 1
    indexed_agents = []
    ratings = []
    for i in rating_data.committee:
        ia: IndexedAgent = rating_data.iagents[i]
        ia.index = ix
        ia.db_id = db_id
        ix += 1
        db_id += 1
        indexed_agents.append(ia)
        ratings.append(rating_data.ratings[i])
        gen = ia.agent.gen
        if gen == 0:
            continue
        src = organizer.get_model_filename(gen)
        shutil.copyfile(src, os.path.join(model_path, f'gen-{gen}.pt'))

    indexed_agents, ratings = zip(*sorted(zip(indexed_agents, ratings), key=lambda x: x[1]))
    cmd = shlex.join(sys.argv)
    RatingDB.save_ratings_to_json(indexed_agents, ratings, file, cmd)
    shutil.copyfile(organizer.binary_filename, os.path.join(path, 'binary'))
    shutil.copyfile(organizer.self_play_db_filename, os.path.join(path, 'self_play.db'))
    shutil.copyfile(organizer.training_db_filename, os.path.join(path, 'training.db'))
    logger.info(f"Created benchmark data folder {path}")
