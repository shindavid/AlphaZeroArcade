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


class Workspace(BaseDir):
    base_dir = '/workspace/mount'
    benchmark_dir = os.path.join(base_dir, 'benchmarks')
    ref_dir = '/workspace/repo/reference.players'
    aws_dir = os.path.join(base_dir, 'aws')
    tars_dir = os.path.join(base_dir, 'tars')

    @staticmethod
    def benchmark_record_file(game: str) -> str:
        return os.path.join('/workspace/repo/benchmark_records', f'{game}.json')

    @staticmethod
    def ref_rundir(game: str) -> str:
        return os.path.join(Workspace.output_dir(), game, 'reference.player')


class Benchmark(BaseDir):
    base_dir = '/workspace/mount'

    @classmethod
    def output_dir(cls):
        return os.path.join(cls.base_dir, 'benchmarks')

    @classmethod
    def tar_path(cls, game: str, tag: str, utc_key: str = None) -> Optional[str]:
        if utc_key is None:
            tag_dir = os.path.join(Workspace.tars_dir, game, tag)
            if not os.path.isdir(tag_dir):
                return None

            tar_files = glob.glob(os.path.join(tag_dir, '*.tar'))
            if tar_files:
                utc_key = max(tar_files)
            else:
                return None
        else:
            utc_key = utc_key + '.tar'
        return os.path.join(Workspace.tars_dir, game, tag, utc_key)

    @classmethod
    def path(cls, game: str, tag: str):
        return os.path.join(cls.output_dir(), game, tag)