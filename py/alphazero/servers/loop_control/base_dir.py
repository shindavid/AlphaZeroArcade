import os

class BaseDir:
    base_dir = ''

    @classmethod
    def output_dir(cls):
        return os.path.join(cls.base_dir, 'output')


class Scratch(BaseDir):
    base_dir = '/home/devuser/scratch'


class Workspace(BaseDir):
    base_dir = '/workspace/mount'
    benchmark_run_dir = os.path.join(base_dir, 'benchmark_runs')
    benchmark_data_dir = os.path.join(base_dir, 'benchmark_data')
    ref_dir = '/workspace/repo/reference.players'

    @staticmethod
    def benchmark_record_file(game: str) -> str:
        return os.path.join('/workspace/repo/benchmark_info', f'{game}.json')
