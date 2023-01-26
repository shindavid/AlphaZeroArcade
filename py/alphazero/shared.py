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
from dataclasses import dataclass
from typing import Any

from natsort import natsorted

from util import subprocess_util
from util.repo_util import Repo


@dataclass
class Param:
    short_name: str
    long_name: str
    value: Any
    help: str

    @property
    def value_type(self):
        return type(self.value)


class OptimizationArgParams:
    """
    AlphaGoZero optimization used:

    - 64 GPU workers
    - 19 CPU parameter servers
    - Minibatch size of 32 per worker (32*64 = 2,048 in total)
    - Minibatches sampled uniformly randomly from most recent 500,000 games
    - Checkpointing every 1,000 training steps
    - Momentum of 0.9
    - L2 regularization parameter of 1e-4
    - Per-sample learning rate annealing from 10^-5 to 10^-7 (from 200k to 600k steps)
    - Unclear how they balanced policy loss vs value loss

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

    KataGo used:

    - 1 GPU
    - Minibatch size of 256
    - Minibatches sampled uniformly randomly from most recent f(N) samples (=positions, not games), where f(N) is
      the function g(x) = x^alpha shifted so that f(c)=c and f'(c)=beta, where (alpha, beta, c) = (0.75, 0.4, 250k)
    - Every ~250k training samples (~1000 training steps), weight snapshot is taken, and EMA of last 4 snapshots
      with decay=0.75 is used as snapshot
    - Momentum of 0.9
    - L2 regularization parameter of 3e-5 (should correspond to weight_decay of 2*3e-5 = 6e-5)
    - Per-sample learning rate annealing from 2*10^-5 (first 5mil) to 6*10^-5, back down to 6*10^-6 for the last day
    - Scaled value loss by 1.5

    https://arxiv.org/pdf/1902.10565.pdf

    TODO: weight EMA
    TODO: learning rate annealing
    """
    minibatch_size = Param('-m', '--minibatch-size', 256, 'minibatch size')
    snapshot_steps = Param('-s', '--snapshot-steps', 1024, 'steps per snapshot')
    window_alpha = Param('-A', '--window-alpha', 0.75, 'alpha for n_window formula')
    window_beta = Param('-B', '--window-beta', 0.4, 'beta for n_window formula')
    window_c = Param('-c', '--window-c', 250000, 'c for n_window formula')
    momentum = Param('-M', '--momentum', 0.9, 'momentum')
    weight_decay = Param('-w', '--weight-decay', 6e-5, 'weight decay')
    learning_rate = Param('-l', '--learning-rate', 6e-5, 'learning rate')


class OptimizationArgs:
    attrs = [attr for attr in dir(OptimizationArgParams) if isinstance(getattr(OptimizationArgParams, attr), Param)]

    minibatch_size: int
    snapshot_steps: int
    window_alpha: float
    window_beta: float
    window_c: int
    momentum: float
    weight_decay: float
    learning_rate: float

    @staticmethod
    def load(args):
        for attr in OptimizationArgs.attrs:
            setattr(OptimizationArgs, attr, getattr(args, attr))

    @staticmethod
    def get_str() -> str:
        tokens = []

        for attr in OptimizationArgs.attrs:
            param: Param = getattr(OptimizationArgParams, attr)
            current = getattr(OptimizationArgs, attr)
            if current != param.value:
                tokens.extend([param.short_name, str(current)])

        return ' '.join(tokens)


def add_optimization_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('alphazero optimization options')

    for attr in OptimizationArgs.attrs:
        param: Param = getattr(OptimizationArgParams, attr)
        group.add_argument(param.short_name, param.long_name, type=param.value_type, default=param.value,
                           help=f'{param.help} (default: %(default)s)')


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
