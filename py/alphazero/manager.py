"""
NOTE: to clearly differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
         current/
             remote.pid
             checkpoint.ptc
             candidate.ptj
             train_and_promote.log
         self-play/
             gen0-{total_positions}/
                 {timestamp}-{num_positions}.ptd
                 ...
             gen1-{total_positions}/
                 ...
             gen2/
                 ...
             ...
         models/
             gen0.ptj
             gen1.ptj
             gen2.ptj
             ...

models/gen{k}.ptj is trained off of self-play/gen{k}/

TODO: make this game-agnostic. There is some hard-coded c4 stuff in here at present.
"""

import os
import shutil
import signal
import subprocess
from typing import Optional, List

from natsort import natsorted

from alphazero.optimization_args import OptimizationArgs
from util import subprocess_util
from util.py_util import timed_print
from util.repo_util import Repo


Generation = int


class AlphaZeroManager:
    managers: List['AlphaZeroManager'] = []
    signal_registered = False

    @staticmethod
    def signal_handler(sig, frame):
        for manager in AlphaZeroManager.managers:
            manager.remotely_kill_pid()

    def __init__(self, c4_base_dir: str):
        AlphaZeroManager.managers.append(self)
        if not AlphaZeroManager.signal_registered:
            AlphaZeroManager.signal_registered = True
            signal.signal(signal.SIGINT, AlphaZeroManager.signal_handler)

        self.c4_base_dir: str = c4_base_dir

        self.models_dir = os.path.join(self.c4_base_dir, 'models')
        self.self_play_dir = os.path.join(self.c4_base_dir, 'self-play')
        self.current_dir = os.path.join(self.c4_base_dir, 'current')

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.self_play_dir, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)

        self.run_index = 0
        self.remote_host = None

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

    def get_latest_games_generation(self) -> Generation:
        self_play_subdirs = list(natsorted(f for f in os.listdir(self.self_play_dir)))
        self_play_subdirs = [f for f in self_play_subdirs if f.startswith('gen')]
        if not self_play_subdirs:
            return -1
        return int(self_play_subdirs[-1][3:].split('-')[0])

    def get_latest_games_dir(self) -> Optional[str]:
        self_play_subdirs = list(natsorted(f for f in os.listdir(self.self_play_dir)))
        self_play_subdirs = [f for f in self_play_subdirs if f.startswith('gen')]
        if not self_play_subdirs:
            return None
        return os.path.join(self.self_play_dir, self_play_subdirs[-1])

    def get_latest_model_generation(self) -> Generation:
        model_files = list(natsorted(f for f in os.listdir(self.models_dir)))
        model_files = [f for f in model_files if f.startswith('gen') and f.endswith('.ptj')]
        if not model_files:
            return -1
        return int(model_files[-1].split('.')[0][3:])

    def get_latest_model_filename(self) -> Optional[str]:
        model_files = list(natsorted(f for f in os.listdir(self.models_dir)))
        model_files = [f for f in model_files if f.startswith('gen') and f.endswith('.ptj')]
        if not model_files:
            return None
        return os.path.join(self.models_dir, model_files[-1])

    def get_pid_filename(self) -> str:
        return os.path.join(self.c4_base_dir, 'current', 'remote.pid')

    def write_pid_file(self):
        pid = os.getpid()
        pid_filename = self.get_pid_filename()
        with open(pid_filename, 'w') as f:
            f.write(str(pid))
        timed_print(f'Wrote pid {pid} to {pid_filename}')

    def remotely_kill_pid(self):
        pid_filename = self.get_pid_filename()
        if not os.path.isfile(pid_filename):
            return
        with open(pid_filename, 'r') as f:
            pid = int(f.readline().strip())

        kill_cmd = f'ssh {self.remote_host} "kill {pid}"'
        timed_print(f'Remotely killing: {kill_cmd}')
        os.system(kill_cmd)

    def remotely_train_model(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        self.remote_host = remote_host
        pid_filename = self.get_pid_filename()
        os.system(f'rm -f {pid_filename}')
        remote_log_filename = os.path.join(remote_c4_base_dir, 'current', 'train_and_promote.log')
        train_cmd = f'python3 -u py/alphazero/train_and_promote.py -d {remote_c4_base_dir} ' + OptimizationArgs.get_str()
        cmd = f'ssh {remote_host} "cd {remote_repo_path}; {train_cmd} |& tee {remote_log_filename}"'
        timed_print(f'Running: {cmd}')
        proc = subprocess_util.Popen(cmd, stdout=None, stderr=None)
        proc.communicate()
        if proc.returncode:
            raise Exception()

        gen = self.get_latest_model_generation()

        candidate = self.get_current_candidate_model_filename()
        promoted = self.get_model_filename(gen + 1)
        shutil.move(candidate, promoted)
        timed_print(f'Promoted {candidate} to {promoted}')

    def run(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        self.run_index += 1
        game_gen = self.get_latest_games_generation()
        model_gen = self.get_latest_model_generation()
        timed_print(f'Running iteration {self.run_index}, game-gen:{game_gen} model-gen:{model_gen}')
        self_play_proc = None
        games_dir = None
        if model_gen >= 0:
            games_dir = self.get_games_dir(model_gen + 1)
            model = self.get_model_filename(model_gen)
            self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_training_self_play')
            self_play_cmd = [
                self_play_bin,
                '-G', '0',
                '-g', games_dir,
                '--nnet-filename', model
            ]
            self_play_proc = subprocess_util.Popen(self_play_cmd, shell=False)
            timed_print(f'Running [{self_play_proc.pid}]: {" ".join(self_play_cmd)}')

        self.remotely_train_model(remote_host, remote_repo_path, remote_c4_base_dir)
        if self_play_proc is not None:
            timed_print(f'Killing self play proc {self_play_proc.pid}...')
            self_play_proc.kill()
            self_play_proc.wait(300)
            timed_print(f'Self play proc killed!')
            n_games = 0
            for filename in os.listdir(games_dir):
                n = int(filename.split('-')[1].split('.')[0])
                n_games += n

            src = games_dir
            tgt = f'{games_dir}-{n_games}'
            os.system(f'mv {src} {tgt}')

    def main_loop(self, remote_host: str, remote_repo_path: str, remote_c4_base_dir: str):
        while True:
            self.run(remote_host, remote_repo_path, remote_c4_base_dir)
