"""
BASE_DIR/
         candidate.pt
         self-play/
               gen0/
               gen1/
               gen2/
               ...
         models/
                gen0.pt
                gen1.pt
                gen2.pt
                ...
"""

import os

from natsort import natsorted

from alphazero.optimization_args import OptimizationArgs
from util import subprocess_util
from util.repo_util import Repo


class AlphaZeroManager:
    def __init__(self, c4_base_dir: str):
        self.c4_base_dir: str = c4_base_dir
        self.self_play_proc = None
        self.models_dir = os.path.join(self.c4_base_dir, 'models')
        self.self_play_dir = os.path.join(self.c4_base_dir, 'self-play')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.self_play_dir, exist_ok=True)

        self.run_index = 0
        self.generation: int = 0

    def init_gen0(self):
        raise Exception('TODO: implement me')

    def load_generation(self):
        model_files = list(natsorted([f for f in os.listdir(self.models_dir) if not f.startswith('.')]))
        if not model_files:
            self.init_gen0()
            self.generation = 0
            return
        last_file = model_files[-1]
        assert last_file.startswith('gen') and last_file.endswith('.pt')
        self.generation = int(last_file[3:].split('.')[0])

    def remotely_train_model(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        # games_dir = os.path.join(self.self_play_dir, f'gen{self.generation}')
        # candidate_model = os.path.join(self.c4_base_dir, 'candidate.pt')
        train_cmd = f'./py/connect4/train_and_promote.py -d {remote_c4_base_dir} ' + OptimizationArgs.get_str()
        cmd = f'ssh {remote_host} "cd {remote_repo_path}; {train_cmd}"'
        print(cmd)
        subprocess_util.run(cmd)
        self.generation += 1

    def run(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        self.run_index += 1
        self.load_generation()
        print(f'Running iteration {self.run_index}, generation {self.generation}')
        games_dir = os.path.join(self.self_play_dir, f'gen{self.generation}')
        model = os.path.join(self.models_dir, f'gen{self.generation}.pt')
        self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_training_self_play')
        self_play_cmd = f'{self_play_bin} -G 0 -g {games_dir} --mcts-nnet-filename {model}'
        print(self_play_cmd)
        self.self_play_proc = subprocess_util.Popen(self_play_cmd)
        self.remotely_train_model(remote_host, remote_repo_path, remote_c4_base_dir)
        self.self_play_proc.kill()

    def main_loop(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        while True:
            self.run(remote_host, remote_repo_path, remote_c4_base_dir)
