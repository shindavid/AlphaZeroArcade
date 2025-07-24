from alphazero.logic.agent_types import IndexedAgent
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Workspace
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.aws_util import BUCKET
from util.py_util import untar_remote_file_to_local_directory

from dataclasses import dataclass
import glob
import logging
import json
import os
import shlex
import shutil
import sys
from typing import Optional


logger = logging.getLogger(__name__)
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

            folders = [f for f in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, f))]
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
        else:
            utc_key = utc_key + '.tar'
        return os.path.join(Workspace.benchmark_data_dir, game, tag, utc_key)


class BenchmarkOption:
    def __init__(self, game: str, tag: Optional[str] = None):
        self.game = game
        self.tag = tag
        self.game_spec: GameSpec = get_game_spec(self.game)

    def has_reference_player(self):
        return self.game_spec.reference_player_family is not None

    def on_record(self) -> Optional[BenchmarkRecord]:
        record: BenchmarkRecord = BenchmarkRecord.load(self.game)
        if self.tag:
            if record and record.tag == self.tag:
                return record
            return None
        elif record:
            return record
        else:
            return None

    def has_run_dir(self) -> bool:
        organizer = DirectoryOrganizer(RunParams(self.game, self.tag), base_dir_root=Workspace)
        return os.path.isdir(organizer.base_dir)

    @staticmethod
    def benchmark_folder(tag: str):
        return f'{tag}.benchmark'

    def has_benchmark_rundir(self) -> bool:
        benchmark_folder = BenchmarkOption.benchmark_folder(self.tag)
        run_params = RunParams(self.game, benchmark_folder)
        organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
        return os.path.isdir(organizer.base_dir)

    def has_benchmark_data(self, utc_key: str = None) -> bool:
        path = BenchmarkData.path(self.game, self.tag, utc_key=utc_key)
        if not path:
            return False
        return os.path.isdir(path)

    def has_benchmark_tar_file(self, utc_key: str = None) -> bool:
        tar_path = BenchmarkData.tar_path(self.game, self.tag, utc_key=utc_key)
        if not tar_path:
            return False
        return os.path.exists(tar_path)

    def has_valid_benchmark(self) -> bool:
        return self.tag or self.on_record() or self.has_reference_player()

    def setup_benchmark_rundir(self) -> Optional[str]:  # benchmark_tag
        if self.tag:
            self.setup_rundir_from_run()
            return self.tag
        elif self.has_reference_player():
            self.setup_rundir_from_reference()
            return 'reference.player'
        else:
            record = self.on_record()
            if record:
                self.setup_rundir_from_record(record)
                return record.tag
            else:
                return None

    def setup_rundir_from_record(self, record: BenchmarkRecord):
        option = BenchmarkOption(record.game, record.tag)
        option.setup_rundir_from_run(utc_key=record.utc_key)

    def setup_rundir_from_reference(self):
        benchmark_organizer = DirectoryOrganizer(RunParams(self.game, 'reference.player.benchmark'))
        self.create_db_from_json(benchmark_organizer, is_reference=True)

    def setup_rundir_from_run(self, utc_key: str = None):
        if self.has_benchmark_rundir():
            logger.debug("benchmark rundir exists.")
            return
        elif self.has_benchmark_data(utc_key=utc_key):
            logger.debug("benchmark data folder exists")
            self.expand_rundir_from_datafolder()
        elif self.has_benchmark_tar_file(utc_key=utc_key):
            logger.debug("benchmark tar file exists")
            self.untar_datafile()
            self.expand_rundir_from_datafolder()
        else:
            logger.debug("read record")
            record = self.on_record()
            if record:
                tar_path = BenchmarkData.tar_path(self.game, self.tag, utc_key=record.utc_key)
                BUCKET.download_from_s3(record.key(), tar_path)
                logger.info(f"File downloaded to {tar_path}")
                self.untar_datafile(utc_key=record.utc_key)
                self.expand_rundir_from_datafolder()
            else:
                raise Exception("no benchmark found when benchmark-tag is specified.")

    def untar_datafile(self, utc_key: str = None):
        tar_path = BenchmarkData.tar_path(self.game, self.tag, utc_key=utc_key)
        untar_remote_file_to_local_directory(tar_path, os.path.dirname(tar_path))
        logger.info(f"untar {tar_path}")

    def expand_rundir_from_datafolder(self, utc_key: str = None):
        assert self.tag is not None
        benchmark_folder = BenchmarkOption.benchmark_folder(self.tag)
        run_params = RunParams(self.game, benchmark_folder)
        benchmark_organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
        benchmark_organizer.dir_setup(benchmark_tag=self.tag)

        self.create_db_from_json(benchmark_organizer, utc_key=utc_key)

        data_folder = BenchmarkData.path(self.game, self.tag, utc_key=utc_key)
        binary = os.path.join(data_folder, 'binary')
        models = os.path.join(data_folder, 'models')
        self_play_db = os.path.join(data_folder, 'self_play.db')
        training_db = os.path.join(data_folder, 'training.db')

        shutil.copyfile(binary, benchmark_organizer.binary_filename)
        shutil.copytree(models, benchmark_organizer.models_dir, dirs_exist_ok=True)
        shutil.copyfile(self_play_db, benchmark_organizer.self_play_db_filename)
        shutil.copyfile(training_db, benchmark_organizer.training_db_filename)
        logger.info(f"copied binary, models, self_play_db and training_db to"
                    f"{benchmark_organizer.base_dir}")

    def create_db_from_json(self, benchmark_organizer, is_reference: bool = False,
                            utc_key: str = None):
        if is_reference:
            json_path = os.path.join(Workspace.ref_dir, f'{self.game}.json')
        else:
            data_folder = BenchmarkData.path(self.game, self.tag, utc_key=utc_key)
            json_path = os.path.join(data_folder, 'ratings.json')
        db = RatingDB(benchmark_organizer.benchmark_db_filename)
        db.load_ratings_from_json(json_path)
        logger.info(f"created db {db.db_filename} from {json_path}")


def save_benchmark_data(organizer: DirectoryOrganizer, record: BenchmarkRecord):
    benchmarker = Benchmarker(organizer)
    path = BenchmarkData.path(record.game, record.tag, utc_key=record.utc_key)
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
