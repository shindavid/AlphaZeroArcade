class BaseDir:
    base_dir = ''
    
    @class_method
    def output_dir(class):
        return os.join.path(class.base_dir, 'output')


class Scratch(BaseDir):
    base_dir = '/home/devuser/scratch'


class Workspace(BaseDir):    
    base_dir = '/workspace/mount'
    output_dir = os.path.join(base_dir, 'output')
    benchmark_run_dir = os.path.join(base_dir, 'benchmark_runs')
    benchmark_data_dir = os.path.join(base_dir, 'benchmark_data')
    ref_dir = '/workspace/repo/reference.players'

    @staticmethod
    def benchmark_info_file(game: str) -> str:
        return os.path.join('/workspace/repo/benchmark_info', f'{game}.json')
