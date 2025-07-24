import logging
import os


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
    benchmark_data_dir = os.path.join(base_dir, 'benchmark_data')
    ref_dir = '/workspace/repo/reference.players'
    aws_dir = os.path.join(base_dir, 'aws')

    @staticmethod
    def benchmark_record_file(game: str) -> str:
        return os.path.join('/workspace/repo/benchmark_records', f'{game}.json')

    @staticmethod
    def ref_rundir(game: str) -> str:
        return os.path.join(Workspace.output_dir(), game, 'reference.player')
