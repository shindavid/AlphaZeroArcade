from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


class BaseDir:
    base_dir = ''

    @classmethod
    def output_dir(cls):
        return os.path.join(cls.base_dir, 'output')


class Scratch(BaseDir):
    base_dir = '/home/devuser/scratch'


@dataclass
class BenchmarkRecord:
    utc_key: Optional[str] = None
    tag: Optional[str] = None
    game: Optional[str] = None
    hash: Optional[str] = None

    def data_folder_path(self):
        utc_key = self.utc_key
        if utc_key is None:
            tag_dir = os.path.join(Workspace.benchmark_data_dir, self.game, self.tag)
            if not os.path.isdir(tag_dir):
                return None

            folders = []
            for name in os.listdir(tag_dir):
                try:
                    dt = datetime.strptime(name, '%Y-%m-%d_%H-%M-%S_UTC')
                    folders.append((dt, name))
                except ValueError:
                    pass

            if folders:
                utc_key = max(folders, key=lambda x: x[0])[1]
            else:
                return None
        return os.path.join(Workspace.benchmark_data_dir, self.game, self.tag, utc_key)

    def to_dict(self):
        return {'utc_key': self.utc_key, 'tag': self.tag, 'hash': self.hash}

    def key(self):
        return os.path.join(self.game, self.tag, f"{self.utc_key}.tar")


class Workspace(BaseDir):
    base_dir = '/workspace/mount'
    benchmark_run_dir = os.path.join(base_dir, 'benchmark_runs')
    benchmark_data_dir = os.path.join(base_dir, 'benchmark_data')
    ref_dir = '/workspace/repo/reference.players'
    aws_dir = os.path.join(base_dir, 'aws')

    @staticmethod
    def benchmark_record_file(game: str) -> str:
        return os.path.join('/workspace/repo/benchmark_records', f'{game}.json')

    @staticmethod
    def load_benchmark_record(game: str) -> Optional[BenchmarkRecord]:
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
        hash = benchmark_record.get("hash", None)

        if utc_key is None or tag is None or hash is None:
            raise ValueError(f"Invalid benchmark info file format for game '{game}': {file_path}")

        return BenchmarkRecord(utc_key=utc_key, tag=tag, game=game, hash=hash)
