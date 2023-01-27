"""
NOTE: to clearly differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
         current/
             checkpoint.ptc
             candidate.ptj
         self-play/
             gen0/
                 {timestamp}-{num_positions}.ptd
                 ...
             gen1/
                 ...
             gen2/
                 ...
             ...
         models/
             gen0.ptj
             gen1.ptj
             gen2.ptj
             ...

TODO: make this game-agnostic. There is some hard-coded c4 stuff in here at present.
"""

import os
import shutil
import subprocess
import sys

from natsort import natsorted

from alphazero.optimization_args import OptimizationArgs
from util import subprocess_util
from util.py_util import timed_print
from util.repo_util import Repo


Generation = int


class AlphaZeroManager:
    def __init__(self, c4_base_dir: str):
        self.c4_base_dir: str = c4_base_dir
        self.self_play_proc = None

        self.models_dir = os.path.join(self.c4_base_dir, 'models')
        self.self_play_dir = os.path.join(self.c4_base_dir, 'self-play')
        self.current_dir = os.path.join(self.c4_base_dir, 'current')

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.self_play_dir, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)

        self.run_index = 0
        self.generation: int = 0

    def init_gen0(self):
        raise Exception('TODO: implement me')

    def get_games_dir(self, gen: Generation) -> str:
        return os.path.join(self.self_play_dir, f'gen{gen}')

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.models_dir, f'gen{gen}.ptj')

    def get_current_candidate_model_filename(self) -> str:
        return os.path.join(self.c4_base_dir, 'current', 'candidate.ptj')

    def get_current_checkpoint_filename(self) -> str:
        return os.path.join(self.c4_base_dir, 'current', 'checkpoint.ptc')

    def load_generation(self):
        self_play_dirs = list(natsorted(f for f in os.listdir(self.self_play_dir)))
        if not self_play_dirs:
            self.init_gen0()
            self.generation = 0
            return
        last_dir = self_play_dirs[-1]
        assert last_dir.startswith('gen'), last_dir
        self.generation = int(last_dir[3:])

    def remotely_train_model(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        train_cmd = f'./py/alphazero/train_and_promote.py -d {remote_c4_base_dir} ' + OptimizationArgs.get_str()
        cmd = f'ssh {remote_host} "cd {remote_repo_path}; {train_cmd}"'
        timed_print(f'Running: {cmd}')
        proc = subprocess_util.Popen(cmd)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            print(stdout)
            print(stderr)
            raise Exception()
        self.generation += 1

        candidate = self.get_current_candidate_model_filename()
        promoted = self.get_model_filename(self.generation)
        shutil.move(candidate, promoted)
        timed_print(f'Promoted {candidate} to {promoted}')

    def run(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        self.run_index += 1
        self.load_generation()
        timed_print(f'Running iteration {self.run_index}, generation {self.generation}')
        gen0 = self.generation == 0
        if not gen0:
            games_dir = self.get_games_dir(self.generation)
            model = self.get_model_filename(self.generation)
            self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_training_self_play')
            self_play_cmd = f'{self_play_bin} -G 0 -g {games_dir} --mcts-nnet-filename {model}'
            timed_print(f'Running: {self_play_cmd}')
            self.self_play_proc = subprocess_util.Popen(self_play_cmd)
        self.remotely_train_model(remote_host, remote_repo_path, remote_c4_base_dir)
        timed_print(f'Killing self play proc...')
        if not gen0:
            self.self_play_proc.kill()
            timed_print(f'Self play proc killed!')

    def main_loop(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        while True:
            self.run(remote_host, remote_repo_path, remote_c4_base_dir)
