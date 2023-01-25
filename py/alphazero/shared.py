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

import argparse
import os

from natsort import natsorted

from util import subprocess_util
from util.repo_util import Repo


class Args:
    window_alpha: float = 0.75
    window_beta: float = 0.4
    window_c: int = 250000
    c4_base_dir: str

    @staticmethod
    def load(args):
        Args.window_alpha = args.window_alpha
        Args.window_beta = args.window_beta
        Args.window_c = args.window_c
        Args.c4_base_dir = args.c4_base_dir


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('alphazero shared options')

    group.add_argument("-A", "--window-alpha", type=float, default=Args.window_alpha,
                       help='alpha for n_window formula (default: %(default)s)')
    group.add_argument("-B", "--window-beta", type=float, default=Args.window_beta,
                       help='beta for n_window formula (default: %(default)s)')
    group.add_argument("-c", "--window-c", type=int, default=Args.window_c,
                       help='beta for n_window formula (default: %(default)s)')
    parser.add_argument("-d", "--c4-base-dir", help='base-dir for gens of games/models')
    parser.add_argument('-r', '--remote-path', help='host:path on which to run model training/promotion')
    parser.add_argument('-D', '--remote-c4-base-dir', help='--c4-base-dir on remote host')


class AlphaZeroManager:
    def __init__(self, c4_base_dir: str):
        self.c4_base_dir: str = c4_base_dir
        self.self_play_proc = None
        self.models_dir = os.path.join(self.c4_base_dir, 'models')
        self.self_play_dir = os.path.join(self.c4_base_dir, 'self-play')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.self_play_dir, exist_ok=True)

        self.generation: int = 0
        self.load_generation()

    def init_gen0(self):
        raise Exception('TODO: implement me')

    def load_generation(self):
        model_files = list(natsorted(os.listdir(self.models_dir)))
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
        train_cmd = f'./py/connect4/train_and_promote.py -d {remote_c4_base_dir}'
        cmd = f'ssh {remote_host} "cd {remote_repo_path}; {train_cmd}"'
        subprocess_util.run(cmd)
        self.generation += 1

    def run(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        games_dir = os.path.join(self.self_play_dir, f'gen{self.generation}')
        model = os.path.join(self.models_dir, f'gen{self.generation}.pt')
        self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_training_self_play')
        self_play_cmd = f'{self_play_bin} -G 0 -g {games_dir} --mcts-nnet-filename {model}'
        self.self_play_proc = subprocess_util.Popen(self_play_cmd)
        self.remotely_train_model(remote_host, remote_repo_path, remote_c4_base_dir)
        self.self_play_proc.kill()

    def main_loop(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        while True:
            self.run(remote_host, remote_repo_path, remote_c4_base_dir)
