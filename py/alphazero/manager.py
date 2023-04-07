"""
NOTE: to clearly differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
         stdout.txt
         self-play-data/
             gen-0/
                 done.txt  # written after gen is complete
                 {timestamp}-{num_positions}.ptd
                 ...
             gen-1/
                 ...
             gen-2/
                 ...
             ...
         models/
             gen-1.ptj
             gen-2.ptj
             ...
         checkpoints/
             gen-1.ptc
             gen-2.ptc
             ...


TODO: make this game-agnostic. There is some hard-coded c4 stuff in here at present.
"""
import os
import shutil
import signal
import sys
import tempfile
import time
from typing import Optional, List

import torch
import torch.nn as nn
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader
from alphazero.optimization_args import ModelingArgs
from connect4.tensorizor import C4Net
from util import subprocess_util
from util.py_util import timed_print, make_hidden_filename
from util.repo_util import Repo
from alphazero.custom_types import Generation
from alphazero.data.games_dataset import GamesDataset

class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class SelfPlayProcData:
    def __init__(self, cmd: str, n_games: int, gen: Generation, games_dir: str):
        self.proc_complete = False
        self.proc = subprocess_util.Popen(cmd)
        self.n_games = n_games
        self.gen = gen
        self.games_dir = games_dir
        timed_print(f'Running gen-{gen} self-play [{self.proc.pid}]: {cmd}')

        if self.n_games:
            self.wait_for_completion()

    def terminate(self, timeout: Optional[int] = None):
        if self.proc_complete:
            return
        self.proc.kill()
        self.wait_for_completion(timeout=timeout, expected_return_code=-int(signal.SIGKILL))

    def wait_for_completion(self, timeout: Optional[int] = None, expected_return_code: int = 0):
        subprocess_util.wait_for(self.proc, timeout=timeout, expected_return_code=expected_return_code)
        AlphaZeroManager.finalize_games_dir(self.games_dir)
        timed_print(f'Completed gen-{self.gen} self-play [{self.proc.pid}]')
        self.proc_complete = True


class AlphaZeroManager:
    def __init__(self, c4_base_dir: str):
        self.py_cuda_device: int = 0
        if not ModelingArgs.synchronous_mode:
            """
            TODO: assert that 2 GPU's are actually available.
            TODO: make this configurable, this is specific to dshin's setup
            """
            self.py_cuda_device = 1
        self.py_cuda_device_str: str = f'cuda:{self.py_cuda_device}'
        self.log_file = None

        self.n_gen0_games = 1000
        self.n_sync_games = 1000
        self.c4_base_dir: str = c4_base_dir

        self._net = None
        self._opt = None

        self.stdout_filename = os.path.join(self.c4_base_dir, 'stdout.txt')
        self.models_dir = os.path.join(self.c4_base_dir, 'models')
        self.checkpoints_dir = os.path.join(self.c4_base_dir, 'checkpoints')
        self.self_play_data_dir = os.path.join(self.c4_base_dir, 'self-play-data')

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)

    def erase_data_after(self, gen: Generation):
        """
        Deletes self-play/ dirs strictly greater than gen, and all models/checkpoints trained off those dirs
        (i.e., models/checkpoints strictly greater than gen+1).
        """
        timed_print(f'Erasing data after gen {gen}')
        g = gen + 1
        while True:
            gen_dir = os.path.join(self.self_play_data_dir, f'gen-{g}')
            if os.path.exists(gen_dir):
                shutil.rmtree(gen_dir, ignore_errors=True)
                g += 1
            else:
                break

        g = gen + 2
        while True:
            model = os.path.join(self.models_dir, f'gen-{g}.ptj')
            checkpoint = os.path.join(self.checkpoints_dir, f'gen-{g}.ptc')
            found = False
            for f in [model, checkpoint]:
                if os.path.exists(f):
                    os.remove(f)
                    found = True
            if not found:
                break
            g += 1

    def get_net_and_optimizer(self, loader: 'DataLoader'):
        if self._net is not None:
            return self._net, self._opt

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            gen = 1
            input_shape = loader.dataset.get_input_shape()
            self._net = C4Net(input_shape)
            timed_print(f'Creating new net with input shape {input_shape}')
        else:
            gen = checkpoint_info.generation
            checkpoint_filename = self.get_checkpoint_filename(gen)
            timed_print(f'Loading checkpoint: {checkpoint_filename}')

            # copying the checkpoint to somewhere local first seems to bypass some sort of filesystem issue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
                shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
                self._net = C4Net.load_checkpoint(tmp_checkpoint_filename)

        self._net.cuda(device=self.py_cuda_device)
        self._net.train()

        learning_rate = ModelingArgs.learning_rate
        momentum = ModelingArgs.momentum
        weight_decay = ModelingArgs.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # TODO: SWA, cycling learning rate

        return self._net, self._opt

    def init_logging(self, filename: str):
        self.log_file = open(filename, 'a')
        sys.stdout = self
        sys.stderr = self

    def write(self, msg):
        sys.__stdout__.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)
        self.flush()

    def flush(self):
        sys.__stdout__.flush()
        if self.log_file is not None:
            self.log_file.flush()

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.models_dir, f'gen-{gen}.ptj')

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self.checkpoints_dir, f'gen-{gen}.ptc')

    def get_self_play_data_subdir(self, gen: Generation) -> str:
        return os.path.join(self.self_play_data_dir, f'gen-{gen}')

    @staticmethod
    def get_ordered_subpaths(path: str) -> List[str]:
        subpaths = list(natsorted(f for f in os.listdir(path)))
        return [f for f in subpaths if not f.startswith('.')]

    @staticmethod
    def get_latest_full_subpath(path: str) -> Optional[str]:
        subpaths = AlphaZeroManager.get_ordered_subpaths(path)
        return os.path.join(path, subpaths[-1]) if subpaths else None

    @staticmethod
    def get_latest_info(path: str) -> Optional[PathInfo]:
        subpaths = AlphaZeroManager.get_ordered_subpaths(path)
        if not subpaths:
            return None
        return PathInfo(subpaths[-1])

    def get_latest_model_info(self) -> Optional[PathInfo]:
        return AlphaZeroManager.get_latest_info(self.models_dir)

    def get_latest_checkpoint_info(self) -> Optional[PathInfo]:
        return AlphaZeroManager.get_latest_info(self.checkpoints_dir)

    def get_latest_model_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.models_dir)
        return 0 if info is None else info.generation

    def get_latest_self_play_data_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.self_play_data_dir)
        return 0 if info is None else info.generation

    def get_latest_model_filename(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.models_dir)

    def get_latest_self_play_data_subdir(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.self_play_data_dir)

    def get_self_play_proc(self, async_mode: bool) -> SelfPlayProcData:
        gen = self.get_latest_model_generation()

        games_dir = self.get_self_play_data_subdir(gen)
        c4_bin = os.path.join(Repo.root(), 'target/Release/bin/c4')
        if gen == 0:
            n_games = self.n_gen0_games
        elif not async_mode:
            n_games = self.n_sync_games
        else:
            n_games = 0

        player_args = [
            '--type=MCTS-T',
            '--no-forced-playouts',
            '-g', games_dir,
        ]

        if gen == 0:
            player_args.append('--uniform-model')
        else:
            model = self.get_model_filename(gen)
            player_args.extend([
                '--nnet-filename', model,
                '--no-clear-dir',
            ])

        self_play_cmd = [
            c4_bin,
            '-G', n_games,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))
        return SelfPlayProcData(self_play_cmd, n_games, gen, games_dir)

    def train_step(self):
        print('******************************')
        gen = self.get_latest_model_generation() + 1
        timed_print(f'Train gen:{gen}')

        value_loss_lambda = ModelingArgs.value_loss_lambda
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.CrossEntropyLoss()

        for_loop_time = 0
        t0 = time.time()
        steps = 0
        while steps < ModelingArgs.snapshot_steps:
            games_dataset = GamesDataset(self.self_play_data_dir)
            loader = torch.utils.data.DataLoader(
                games_dataset,
                batch_size = ModelingArgs.minibatch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True)
            assert games_dataset.n_total_games >= self.n_gen0_games

            net, optimizer = self.get_net_and_optimizer(loader)

            timed_print(f'Sampling from the {games_dataset.n_window} most recent positions among '
                        f'{games_dataset.n_total_positions} total positions (minibatches processed: {steps})')

            stats = TrainingStats()
            for data in loader:
                t1 = time.time()
                inputs, value_labels, policy_labels = data
                inputs = inputs.to(self.py_cuda_device_str)
                value_labels = value_labels.to(self.py_cuda_device_str)
                policy_labels = policy_labels.to(self.py_cuda_device_str)

                optimizer.zero_grad()
                policy_outputs, value_outputs = net(inputs)
                policy_loss = policy_criterion(policy_outputs, policy_labels)
                value_loss = value_criterion(value_outputs, value_labels)
                loss = policy_loss + value_loss * value_loss_lambda

                stats.update(policy_labels, policy_outputs, policy_loss, value_labels, value_outputs, value_loss)

                loss.backward()
                optimizer.step()
                steps += 1
                t2 = time.time()
                for_loop_time += t2 - t1
                if steps == ModelingArgs.snapshot_steps:
                    break
            stats.dump()

        t3 = time.time()
        total_time = t3 - t0
        data_loading_time = total_time - for_loop_time

        timed_print(f'Gen {gen} training complete ({steps} minibatch updates)')
        timed_print(f'Data loading time: {data_loading_time:10.3f} seconds')
        timed_print(f'Training time:     {for_loop_time:10.3f} seconds')

        checkpoint_filename = self.get_checkpoint_filename(gen)
        model_filename = self.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        net.save_checkpoint(tmp_checkpoint_filename)
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        timed_print(f'Checkpoint saved: {checkpoint_filename}')
        timed_print(f'Model saved: {model_filename}')

    @staticmethod
    def finalize_games_dir(games_dir: str):
        n_positions = 0
        n_games = 0
        for filename in os.listdir(games_dir):
            n = int(filename.split('-')[1].split('.')[0])
            n_positions += n
            n_games += 1

        done_file = os.path.join(games_dir, 'done.txt')
        with open(done_file, 'w') as f:
            f.write(f'n_games={n_games}\n')
            f.write(f'n_positions={n_positions}\n')
            f.write(f'done\n')

    def run(self, async_mode: bool = True):
        self.init_logging(self.stdout_filename)

        while True:
            self_play_proc_data = self.get_self_play_proc(async_mode)
            self.train_step()
            self_play_proc_data.terminate(timeout=300)


def get_num_correct_policy_predictions(policy_outputs, policy_labels):
    selected_moves = torch.argmax(policy_outputs, dim=1)
    correct_policy_preds = policy_labels.gather(1, selected_moves.view(-1, 1))
    return int(sum(correct_policy_preds))


def get_num_correct_value_predictions(value_outputs, value_labels):
    value_output_probs = value_outputs.softmax(dim=1)
    deltas = abs(value_output_probs - value_labels)
    return int(sum((deltas < 0.25).all(dim=1)))


class TrainingStats:
    def __init__(self):
        self.policy_accuracy_num = 0.0
        self.policy_loss_num = 0.0
        self.value_accuracy_num = 0.0
        self.value_loss_num = 0.0
        self.den = 0

    def update(self, policy_labels, policy_outputs, policy_loss, value_labels, value_outputs, value_loss):
        n = len(policy_labels)
        self.policy_loss_num += float(policy_loss.item()) * n
        self.value_loss_num += float(value_loss.item()) * n
        self.den += n
        self.policy_accuracy_num += get_num_correct_policy_predictions(policy_outputs, policy_labels)
        self.value_accuracy_num += get_num_correct_value_predictions(value_outputs, value_labels)

    def dump(self):
        policy_accuracy = self.policy_accuracy_num / self.den
        avg_policy_loss = self.policy_loss_num / self.den
        value_accuracy = self.value_accuracy_num / self.den
        avg_value_loss = self.value_loss_num / self.den

        print(f'Policy accuracy: %8.6f' % policy_accuracy)
        print(f'Policy loss:     %8.6f' % avg_policy_loss)
        print(f'Value accuracy:  %8.6f' % value_accuracy)
        print(f'Value loss:      %8.6f' % avg_value_loss)
