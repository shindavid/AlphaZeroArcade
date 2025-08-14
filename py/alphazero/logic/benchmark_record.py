"""
The benchmark directory structure is as follows:

benchmarks/
├── {game}/
│   ├── {tag}/
│   │   ├── bin/{game} (game binary)
│   │   ├── databases/
│   │   │   ├── benchmark.db
│   │   │   ├── self_play.db
│   │   │   └── training.db
│   │   ├── misc/version_file
│   │   ├── models/ (only the models of committee agents)
|   │   |   ├── gen-7.onnx
|   │   |   ├── gen-14.onnx
|   │   |   └── ...
|   │   └── timing_caches/
│   │       ├── e7980f02e07c42bbaaeb0a80a64c51bd  # model architecture hash
│   │       |   ├── sm_7.9_trt25.06_cuda12.9.cache
│   │       |   ├── sm_8.0_trt25.06_cuda12.9.cache
│   │       │   └── ...
│   │       └── ...
|   └── ...
└── ...

A benchmark directory is created by running the script `benchmark_tag_local.py`.
"""

from alphazero.logic.agent_types import IndexedAgent
from alphazero.logic.custom_types import Version
from alphazero.logic.self_evaluator import BenchmarkRatingData, SelfEvaluator
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Benchmark, Workspace, VERSION
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.aws_util import BUCKET
from util.py_util import untar_remote_file_to_local_directory

from dataclasses import dataclass
import logging
import json
import os
import shutil
from typing import Optional


logger = logging.getLogger(__name__)
UTC_FORMAT = '%Y-%m-%d_%H-%M-%S.%f_UTC'
BenchmarkTag = str


@dataclass
class BenchmarkRecord:
    utc_key: Optional[str] = None
    tag: Optional[str] = None
    game: Optional[str] = None
    version: Version = VERSION

    def to_dict(self):
        return {'utc_key': self.utc_key, 'tag': self.tag, 'version': self.version.num}

    def key(self):
        return os.path.join(str(self.version), self.game, self.tag, f"{self.utc_key}.tar")

    @staticmethod
    def load(game: str) -> Optional['BenchmarkRecord']:
        """
        This will read the file:
            /workspace/repo/benchmark_records/v{version}/{game}.json
        """

        file_path = Workspace.benchmark_record_file(game)

        if not os.path.exists(file_path):
            logger.info(f"No benchmark record found for game '{game}' at {file_path}. ")
            return None

        with open(file_path, 'r') as f:
            benchmark_record = json.load(f)

        utc_key = benchmark_record["utc_key"]
        tag = benchmark_record["tag"]
        version = Version(num=benchmark_record["version"])

        if utc_key is None or tag is None:
            raise ValueError(f"Invalid benchmark info file format for game '{game}': {file_path}")

        return BenchmarkRecord(utc_key=utc_key, tag=tag, game=game, version=version)

    def version_matches(self) -> bool:
        return self.version == VERSION

class BenchmarkDir:
    @staticmethod
    def path(game: str, tag: str, utc_key: str = None) -> Optional[str]:
        if utc_key is None:
            tag_dir = os.path.join(Workspace.benchmark_dir, game, tag)
            if not os.path.isdir(tag_dir):
                return None

            folders = [f for f in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, f))]
            if folders:
                utc_key = max(folders)
            else:
                return None
        return os.path.join(Workspace.benchmark_dir, game, tag, utc_key)


class BenchmarkData:
    def __init__(self, game: str, tag: Optional[str] = None):
        self.game = game
        self.tag = tag
        self.game_spec: GameSpec = get_game_spec(self.game)

    def setup_rundir(self) -> BenchmarkTag:
        if self.tag:
            self._setup_rundir_from_run()
            return self.tag
        elif self._reference_player_exists():
            self._setup_rundir_from_reference()
            return 'reference.player'
        else:
            record = self._load_record()
            if record:
                self._setup_rundir_from_record(record)
                return record.tag
            else:
                raise Exception("Failed to set up a valid benchmark")

    def valid(self) -> bool:
        return self.tag or self._load_record() or self._reference_player_exists()

    def _create_db_from_json(self, benchmark_organizer: DirectoryOrganizer):
        db = RatingDB(benchmark_organizer.benchmark_db_filename)
        if os.path.exists(db.db_filename) and not db.is_empty():
            logger.debug(f"{benchmark_organizer.benchmark_db_filename} exists."
                         f"Skip loading from json.")
            return

        json_path = os.path.join(Workspace.ref_dir, f'{self.game}.json')
        db.load_ratings_from_json(json_path)
        logger.info(f"created db {db.db_filename} from {json_path}")

    def _load_record(self) -> Optional[BenchmarkRecord]:
        record: BenchmarkRecord = BenchmarkRecord.load(self.game)
        if not record.version_matches():
            logger.warning(f"Benchmark record version mismatch: record version"
                           f"{record.version} != code version {VERSION}.")
            return None

        if self.tag:
            if record and record.tag == self.tag:
                return record
            return None
        elif record:
            return record
        else:
            return None

    def _reference_player_exists(self):
        return self.game_spec.reference_player_family is not None

    def _rundir_exists(self) -> bool:
        return os.path.isdir(Benchmark.path(self.game, self.tag))

    def _setup_rundir_from_record(self, record: BenchmarkRecord):
        benchmark_data = BenchmarkData(record.game, record.tag)
        benchmark_data._setup_rundir_from_run(utc_key=record.utc_key)

    def _setup_rundir_from_reference(self):
        ref_tag = 'reference.player'
        run_params = RunParams(self.game, ref_tag)
        dst_organizer = DirectoryOrganizer(run_params, base_dir_root=Benchmark)
        dst_organizer.dir_setup(benchmark_tag=ref_tag)
        self._create_db_from_json(dst_organizer)

    def _setup_rundir_from_run(self, utc_key: str = None):
        if self._rundir_exists():
            logger.debug("benchmark rundir exists.")
            return
        elif self._tar_file_exists(utc_key=utc_key):
            logger.debug("benchmark tar file exists")
            self._untar()
        else:
            logger.debug("read record")
            record = self._load_record()
            if record:
                tar_path = Benchmark.tar_path(self.game, self.tag, utc_key=record.utc_key)
                BUCKET.download_from_s3(record.key(), tar_path)
                logger.info(f"File downloaded to {tar_path}")
                self._untar(utc_key=record.utc_key)
            else:
                raise Exception("no benchmark found when benchmark-tag is specified.")

    def _tar_file_exists(self, utc_key: str = None) -> bool:
        tar_path = Benchmark.tar_path(self.game, self.tag, utc_key=utc_key)
        if not tar_path:
            return False
        return os.path.exists(tar_path)

    def _untar(self, utc_key: str = None):
        tar_path = Benchmark.tar_path(self.game, self.tag, utc_key=utc_key)
        benchmark_path = Benchmark.game_dir(self.game)
        untar_remote_file_to_local_directory(tar_path, benchmark_path)
        logger.info(f"untar {tar_path} to {benchmark_path}")


def save_benchmark_dir(organizer: DirectoryOrganizer):
    dst_organizer = DirectoryOrganizer(organizer.args, base_dir_root=Benchmark)
    dst_organizer.dir_setup(benchmark_tag=organizer.args.tag)

    self_evaluator = SelfEvaluator(organizer)
    rating_data: BenchmarkRatingData = self_evaluator.read_ratings_from_db()

    for i in rating_data.committee:
        ia: IndexedAgent = rating_data.iagents[i]
        gen = ia.agent.gen
        if gen == 0:
            continue
        src = organizer.get_model_filename(gen)
        shutil.copyfile(src, dst_organizer.get_model_filename(gen))

    shutil.copyfile(organizer.benchmark_db_filename, dst_organizer.benchmark_db_filename)
    shutil.copyfile(organizer.binary_filename, dst_organizer.binary_filename)
    shutil.copyfile(organizer.self_play_db_filename, dst_organizer.self_play_db_filename)
    shutil.copyfile(organizer.training_db_filename, dst_organizer.training_db_filename)
    logger.info(f"Created benchmark data folder {dst_organizer.base_dir}")
