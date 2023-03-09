"""
NOTE: to clearly differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
         kill-file.txt
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
         candidate-models/
             gen-1-epoch-1.ptj
             gen-1-epoch-2.ptj
             gen-2-epoch-1.ptj
             ...
         checkpoints/
             gen-1-epoch-1.ptc
             gen-1-epoch-2.ptc
             gen-2-epoch-1.ptc
             ...
        promoted-models/
             gen-1.ptj
             gen-2.ptj
             ...
        gating_logs/
             gen-1-epoch-1.txt
             gen-1-epoch-2.txt
             ...
        stdouts/
             self-play.txt
             train.txt
             promote.txt


TODO: make this game-agnostic. There is some hard-coded c4 stuff in here at present.
"""

import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch import optim
from natsort import natsorted

from alphazero.optimization_args import ModelingArgs, GatingArgs
from connect4.tensorizor import C4Net
from util import subprocess_util
from util.py_util import timed_print
from util.repo_util import Repo
from util.torch_util import Shape

Generation = int
Epoch = int


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1
        self.epoch: Epoch = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])
            elif token == 'epoch':
                self.epoch = int(tokens[t+1])


class AlphaZeroManager:
    def __init__(self, c4_base_dir: str):
        self.py_cuda_device: int = 1  # TODO: make this configurable, this is specific to dshin's setup
        self.py_cuda_device_str: str = f'cuda:{self.py_cuda_device}'
        self.log_file = None

        self.n_gen0_games = 1000
        self.c4_base_dir: str = c4_base_dir
        self.silence_promote_skip_msgs = False
        self.last_tested_candidate_model_gen_epoch = (-1, -1)
        self.ran_gen0_self_play = False

        self.kill_filename = os.path.join(self.c4_base_dir, 'kill-file.txt')
        self.candidate_models_dir = os.path.join(self.c4_base_dir, 'candidate-models')
        self.checkpoints_dir = os.path.join(self.c4_base_dir, 'checkpoints')
        self.promoted_models_dir = os.path.join(self.c4_base_dir, 'promoted-models')
        self.self_play_data_dir = os.path.join(self.c4_base_dir, 'self-play-data')
        self.gating_logs_dir = os.path.join(self.c4_base_dir, 'gating-logs')
        self.stdouts_dir = os.path.join(self.c4_base_dir, 'stdouts')

        os.makedirs(self.candidate_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.promoted_models_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)
        os.makedirs(self.gating_logs_dir, exist_ok=True)
        os.makedirs(self.stdouts_dir, exist_ok=True)

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

    def kill_file_exists(self):
        return os.path.exists(self.kill_filename)

    def rm_kill_file(self):
        if os.path.exists(self.kill_filename):
            os.remove(self.kill_filename)

    def touch_kill_file(self):
        with open(self.kill_filename, 'w') as _:
            pass

    def get_candidate_model_filename(self, gen: Generation, epoch: Epoch) -> str:
        return os.path.join(self.candidate_models_dir, f'gen-{gen}-epoch-{epoch}.ptj')

    def get_checkpoint_filename(self, gen: Generation, epoch: Epoch) -> str:
        return os.path.join(self.checkpoints_dir, f'gen-{gen}-epoch-{epoch}.ptc')

    def get_promoted_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.promoted_models_dir, f'gen-{gen}.ptj')

    def get_self_play_data_subdir(self, gen: Generation) -> str:
        return os.path.join(self.self_play_data_dir, f'gen-{gen}')

    def get_gating_log_filename(self, gen: Generation, epoch: Epoch) -> str:
        return os.path.join(self.gating_logs_dir, f'gen-{gen}-epoch-{epoch}.txt')

    def get_self_play_stdout(self) -> str:
        return os.path.join(self.stdouts_dir, 'self-play.txt')

    def get_train_stdout(self) -> str:
        return os.path.join(self.stdouts_dir, 'train.txt')

    def get_promote_stdout(self) -> str:
        return os.path.join(self.stdouts_dir, 'promote.txt')

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

    def get_latest_candidate_model_info(self) -> Optional[PathInfo]:
        return AlphaZeroManager.get_latest_info(self.candidate_models_dir)

    def get_latest_checkpoint_info(self) -> Optional[PathInfo]:
        return AlphaZeroManager.get_latest_info(self.checkpoints_dir)

    def get_latest_promoted_model_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.promoted_models_dir)
        return 0 if info is None else info.generation

    def get_latest_self_play_data_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.self_play_data_dir)
        return 0 if info is None else info.generation

    def get_latest_candidate_model_filename(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.candidate_models_dir)

    def get_latest_promoted_model_filename(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.promoted_models_dir)

    def get_latest_self_play_data_subdir(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.self_play_data_dir)

    def self_play(self):
        game_gen = self.get_latest_self_play_data_generation()
        model_gen = self.get_latest_promoted_model_generation()
        gen0 = model_gen == 0
        if (game_gen > 0 >= model_gen) or (gen0 and self.ran_gen0_self_play):
            timed_print('No model to use for self-play.  Waiting for model to be promoted...')
            time.sleep(5)
            return

        self.ran_gen0_self_play = True
        timed_print(f'Running self-play game-gen:{game_gen} model-gen:{model_gen}')
        games_dir = self.get_self_play_data_subdir(model_gen)
        model = self.get_promoted_model_filename(model_gen)
        self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_training_self_play')
        self_play_cmd = [
            self_play_bin,
            '-g', games_dir,
        ]
        if gen0:
            self_play_cmd.extend([
                '-G', self.n_gen0_games,
                '--uniform-model',
            ])
        else:
            self_play_cmd.extend([
                '-G', 0,
                '--nnet-filename', model,
                '--no-clear-dir',
            ])

        self_play_cmd = list(map(str, self_play_cmd))
        self_play_proc = subprocess_util.Popen(self_play_cmd)
        timed_print(f'Running [{self_play_proc.pid}]: {" ".join(self_play_cmd)}')
        if gen0:
            subprocess_util.wait_for(self_play_proc)
        else:
            timed_print(f'Looping until model promotion ({model_gen})...')
            while True:
                cur_model_gen = self.get_latest_promoted_model_generation()
                if cur_model_gen <= model_gen:
                    time.sleep(5)
                    continue
                break

            timed_print(f'Detected model promotion ({model_gen} -> {cur_model_gen})!')
            timed_print(f'Killing self play proc {self_play_proc.pid}...')
            self_play_proc.kill()
            self_play_proc.wait(300)
            timed_print(f'Self play proc killed!')

        AlphaZeroManager.finalize_games_dir(games_dir)

    def train(self):
        print('******************************')
        loader = DataLoader(self.self_play_data_dir)
        if loader.n_total_games < self.n_gen0_games:
            timed_print(f'Not enough games to train: {loader.n_total_games} < {self.n_gen0_games}')
            timed_print('Waiting for more games...')
            time.sleep(5)
            return

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            gen, epoch = 1, 0
            input_shape = loader.get_input_shape()
            net = C4Net(input_shape)
            timed_print(f'Creating new net with input shape {input_shape}')
            timed_print(f'Train gen:{gen} epoch:{epoch + 1}')
        else:
            gen, epoch = checkpoint_info.generation, checkpoint_info.epoch
            checkpoint_filename = self.get_checkpoint_filename(gen, epoch)
            timed_print(f'Loading checkpoint: {checkpoint_filename}')

            # copying the checkpoint to somewhere local first seems to bypass some sort of filesystem issue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
                shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
                net = C4Net.load_checkpoint(tmp_checkpoint_filename)

            latest_promoted_gen = self.get_latest_promoted_model_generation()
            if latest_promoted_gen >= gen:
                gen = latest_promoted_gen + 1
                epoch = 0

            timed_print(f'Train gen:{gen} epoch:{epoch + 1}')

        net.cuda(device=self.py_cuda_device)
        net.train()

        value_loss_lambda = ModelingArgs.value_loss_lambda
        learning_rate = ModelingArgs.learning_rate
        momentum = ModelingArgs.momentum
        weight_decay = ModelingArgs.weight_decay
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        policy_criterion = nn.MultiLabelSoftMarginLoss()
        value_criterion = nn.CrossEntropyLoss()

        timed_print(f'Sampling from the {loader.n_window} most recent positions among '
                    f'{loader.n_total_positions} total positions')

        stats = TrainingStats()

        # TODO: more efficient data loading via pytorch DataLoader
        for i, data in enumerate(loader):
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

        timed_print(f'Gen {gen} epoch {epoch} complete')
        stats.dump()

        checkpoint_filename = self.get_checkpoint_filename(gen, epoch + 1)
        candidate_filename = self.get_candidate_model_filename(gen, epoch + 1)
        net.save_checkpoint(checkpoint_filename)
        net.save_model(candidate_filename)
        timed_print(f'Checkpoint saved: {checkpoint_filename}')
        timed_print(f'Candidate saved: {candidate_filename}')

    def promote(self):
        candidate_model_info = self.get_latest_candidate_model_info()

        if candidate_model_info is None:
            # candidate_model_epoch = self.get_latest_candidate_model_epoch()
            if not self.silence_promote_skip_msgs:
                timed_print(f'No candidate models available. Waiting...')
            time.sleep(5)
            self.silence_promote_skip_msgs = True
            return

        gen, epoch = candidate_model_info.generation, candidate_model_info.epoch
        if (gen, epoch) <= self.last_tested_candidate_model_gen_epoch:
            if not self.silence_promote_skip_msgs:
                timed_print(f'Latest candidate was already tested ({gen}, {epoch})')
            time.sleep(5)
            self.silence_promote_skip_msgs = True
            return

        self.silence_promote_skip_msgs = False
        self.last_tested_candidate_model_gen_epoch = (gen, epoch)

        candidate_model_filename = self.get_candidate_model_filename(gen, epoch)
        latest_promoted_model_filename = self.get_latest_promoted_model_filename()

        if latest_promoted_model_filename is None:
            timed_print(f'First promotion test: auto-pass!')
            promote = True
        else:
            gating_log_filename = self.get_gating_log_filename(gen, epoch)

            self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_competitive_self_play')
            n_games = GatingArgs.num_games
            args = [
                self_play_bin,
                '-G', n_games,
                '-i', GatingArgs.mcts_iters,
                '--batch-size-limit', 64,  # appears best on dshin laptop based on ad-hoc testing
                '-p', 50,  # appears best on dshin laptop based on ad-hoc testing
                '--nnet-filename', latest_promoted_model_filename,
                '--nnet-filename2', candidate_model_filename,
                '--grade-moves',
            ]
            cmd = ' '.join(map(str, args))
            cmd = f'{cmd} > {gating_log_filename}'
            timed_print(f'Running: {cmd}')
            subprocess_util.run(cmd)

            with open(gating_log_filename, 'r') as f:
                stdout = f.read()

            win_rate = extract_win_score(stdout, 1) / n_games
            promote = win_rate > GatingArgs.promotion_win_rate
            timed_print('Run complete.')
            print(f'Candidate win-rate: %.5f' % win_rate)
            print(f'Promotion win-rate: %.5f' % GatingArgs.promotion_win_rate)
            print(f'Promote: %s' % promote)

        if promote:
            src = candidate_model_filename
            dst = self.get_promoted_model_filename(gen)
            timed_print(f'Promotion: {src} -> {dst}')
            shutil.copy(src, dst)

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

    def self_play_loop(self):
        try:
            self.init_logging(self.get_self_play_stdout())
            print('')
            while not self.kill_file_exists():
                self.self_play()
        except (Exception,):
            traceback.print_exc()
            self.touch_kill_file()
            sys.exit(1)

    def train_loop(self):
        try:
            self.init_logging(self.get_train_stdout())
            print('')
            while not self.kill_file_exists():
                self.train()
        except (Exception,):
            traceback.print_exc()
            self.touch_kill_file()
            sys.exit(1)

    def promote_loop(self):
        try:
            self.init_logging(self.get_promote_stdout())
            print('')
            while not self.kill_file_exists():
                self.promote()
        except (Exception,):
            traceback.print_exc()
            self.touch_kill_file()
            sys.exit(1)


class SelfPlayGameMetadata:
    def __init__(self, filename: str):
        self.filename = filename
        info = os.path.split(filename)[1].split('.')[0].split('-')  # 1685860410604914-10.ptd
        self.timestamp = int(info[0])
        self.n_positions = int(info[1])


class GenerationMetadata:
    def __init__(self, full_gen_dir: str):
        self._loaded = False
        self.full_gen_dir = full_gen_dir
        self._game_metadata_list = []

        done_file = os.path.join(full_gen_dir, 'done.txt')
        if os.path.isfile(done_file):
            with open(done_file, 'r') as f:
                lines = list(f.readlines())

            if len(lines) >= 3:
                assert lines[0].startswith('n_games='), lines
                assert lines[1].startswith('n_positions='), lines
                self.n_games = int(lines[0].split('=')[1].strip())
                self.n_positions = int(lines[1].split('=')[1].strip())
                return

        self.n_positions = 0
        self.n_games = 0
        self.load()

    @property
    def game_metadata_list(self):
        self.load()
        return self._game_metadata_list

    def load(self):
        if self._loaded:
            return

        self._loaded = True
        for filename in os.listdir(self.full_gen_dir):
            if filename.startswith('.') or filename == 'done.txt':
                continue
            full_filename = os.path.join(self.full_gen_dir, filename)
            game_metadata = SelfPlayGameMetadata(full_filename)
            self._game_metadata_list.append(game_metadata)

        self._game_metadata_list.sort(key=lambda g: -g.timestamp)  # newest to oldest
        self.n_positions = sum(g.n_positions for g in self._game_metadata_list)
        self.n_games = len(self._game_metadata_list)


class SelfPlayMetadata:
    def __init__(self, self_play_dir: str):
        self.self_play_dir = self_play_dir
        self.metadata: Dict[Generation, GenerationMetadata] = {}
        self.n_total_positions = 0
        self.n_total_games = 0
        for gen_dir in os.listdir(self_play_dir):
            assert gen_dir.startswith('gen-'), gen_dir
            generation = int(gen_dir.split('-')[1])
            full_gen_dir = os.path.join(self_play_dir, gen_dir)
            metadata = GenerationMetadata(full_gen_dir)
            self.metadata[generation] = metadata
            self.n_total_positions += metadata.n_positions
            self.n_total_games += metadata.n_games

    def get_window(self, n_window: int) -> List[SelfPlayGameMetadata]:
        window = []
        cumulative_n_positions = 0
        for generation in reversed(sorted(self.metadata.keys())):  # newest to oldest
            gen_metadata = self.metadata[generation]
            n = len(gen_metadata.game_metadata_list)
            i = 0
            while cumulative_n_positions < n_window and i < n:
                game_metadata = gen_metadata.game_metadata_list[i]
                cumulative_n_positions += game_metadata.n_positions
                i += 1
                window.append(game_metadata)
        return window


class DataLoader:
    def __init__(self, self_play_data_dir: str):
        self.self_play_metadata = SelfPlayMetadata(self_play_data_dir)
        self.n_total_games = self.self_play_metadata.n_total_games
        self.n_total_positions = self.self_play_metadata.n_total_positions
        self.n_window = compute_n_window(self.n_total_positions)
        self.window = self.self_play_metadata.get_window(self.n_window)

        self._returned_snapshots = 0
        self._index = len(self.window)

    def get_input_shape(self) -> Shape:
        for game_metadata in self.window:
            data = torch.jit.load(game_metadata.filename).state_dict()
            return data['input'].shape[1:]
        raise Exception('Could not determine input shape!')

    def __iter__(self):
        return self

    def __next__(self):
        if self._returned_snapshots == ModelingArgs.snapshot_steps:
            raise StopIteration

        self._returned_snapshots += 1
        minibatch: List[SelfPlayGameMetadata] = []
        n = 0
        while n < ModelingArgs.minibatch_size:
            n += self._add_to_minibatch(minibatch)

        input_data = []
        policy_data = []
        value_data = []

        for metadata in minibatch:
            data = torch.jit.load(metadata.filename).state_dict()
            input_data.append(data['input'])
            policy_data.append(data['policy'])
            value_data.append(data['value'])

        input_data = torch.concat(input_data)
        policy_data = torch.concat(policy_data)
        value_data = torch.concat(value_data)

        return input_data, value_data, policy_data

    def _add_to_minibatch(self, minibatch: List[SelfPlayGameMetadata]):
        if self._index == len(self.window):
            random.shuffle(self.window)
            self._index = 0

        game_metadata = self.window[self._index]
        minibatch.append(game_metadata)
        self._index += 1
        return game_metadata.n_positions


def compute_n_window(n_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    c = ModelingArgs.window_c
    alpha = ModelingArgs.window_alpha
    beta = ModelingArgs.window_beta
    return min(n_total, int(c * (1 + beta * ((n_total / c) ** alpha - 1) / alpha)))


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

        print(f'Policy accuracy: %5.3f' % policy_accuracy)
        print(f'Policy loss:     %5.3f' % avg_policy_loss)
        print(f'Value accuracy:  %5.3f' % value_accuracy)
        print(f'Value loss:      %5.3f' % avg_value_loss)


def extract_win_score(stdout: str, player_index: int):
    # P1 W69 L9 D22 [80]
    player_str = f'P{player_index}'
    lines = [line for line in stdout.splitlines() if line.startswith(player_str)]
    assert len(lines) == 1, stdout
    perf_line = lines[0]
    win_score_token = perf_line.split()[-1]
    assert win_score_token.startswith('[') and win_score_token.endswith(']'), perf_line
    return float(win_score_token[1:-1])
