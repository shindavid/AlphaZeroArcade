#!/usr/bin/env python3

"""
TODO: make this game-agnostic. There is some hard-coded c4 stuff in here at present.
"""
import argparse
import os
import random
import shutil
import tempfile
from typing import Dict, List

import torch
import torch.nn as nn
from torch import optim

from alphazero.manager import AlphaZeroManager, Generation
from alphazero.optimization_args import add_optimization_args, OptimizationArgs, ModelingArgs, GatingArgs
from config import Config
from connect4.tensorizor import C4Net
from neural_net import NeuralNet
from util import subprocess_util
from util.py_util import timed_print
from util.repo_util import Repo


class Args:
    c4_base_dir: str

    @staticmethod
    def load(args):
        Args.c4_base_dir = args.c4_base_dir
        assert Args.c4_base_dir, 'Required option: -d'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()
    cfg.add_parser_argument('c4.base_dir', parser, '-d', '--c4-base-dir', help='base-dir for game/model files')
    add_optimization_args(parser)

    args = parser.parse_args()
    Args.load(args)
    OptimizationArgs.load(args)


class SelfPlayGameMetadata:
    def __init__(self, filename: str):
        self.filename = filename
        info = os.path.split(filename)[1].split('.')[0].split('-')  # 1685860410604914-10.ptd
        self.timestamp = int(info[0])
        self.num_positions = int(info[1])


class GenerationMetadata:
    def __init__(self, full_gen_dir: str):
        self._loaded = False
        self.full_gen_dir = full_gen_dir
        self._game_metadata_list = []
        self.num_positions = 0

        gen_subdir = os.path.split(full_gen_dir)[1]  # gen3 or gen3-1234
        tokens = gen_subdir.split('-')
        if len(tokens) == 1:
            self.load()
        else:
            self.num_positions = int(tokens[1])

    @property
    def game_metadata_list(self):
        self.load()
        return self._game_metadata_list

    def load(self):
        if self._loaded:
            return

        self._loaded = True
        for filename in os.listdir(self.full_gen_dir):
            if filename.startswith('.'):
                continue
            full_filename = os.path.join(self.full_gen_dir, filename)
            game_metadata = SelfPlayGameMetadata(full_filename)
            self._game_metadata_list.append(game_metadata)

        self._game_metadata_list.sort(key=lambda g: -g.timestamp)  # newest to oldest
        self.num_positions = sum(g.num_positions for g in self._game_metadata_list)


def compute_n_window(N_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    c = ModelingArgs.window_c
    alpha = ModelingArgs.window_alpha
    beta = ModelingArgs.window_beta
    return min(N_total, int(c * (1 + beta * ((N_total / c) ** alpha - 1) / alpha)))


class SelfPlayMetadata:
    def __init__(self, self_play_dir: str):
        self.self_play_dir = self_play_dir
        self.metadata: Dict[Generation, GenerationMetadata] = {}
        self.n_total_positions = 0
        for gen_dir in os.listdir(self_play_dir):
            assert gen_dir.startswith('gen'), gen_dir
            generation = int(gen_dir[3:].split('-')[0])
            full_gen_dir = os.path.join(self_play_dir, gen_dir)
            metadata = GenerationMetadata(full_gen_dir)
            self.metadata[generation] = metadata
            self.n_total_positions += metadata.num_positions

    def get_window(self, n_window: int) -> List[SelfPlayGameMetadata]:
        window = []
        cumulative_n_positions = 0
        for generation in reversed(sorted(self.metadata.keys())):  # newest to oldest
            gen_metadata = self.metadata[generation]
            n = len(gen_metadata.game_metadata_list)
            i = 0
            while cumulative_n_positions < n_window and i < n:
                game_metadata = gen_metadata.game_metadata_list[i]
                cumulative_n_positions += game_metadata.num_positions
                i += 1
                window.append(game_metadata)
        return window


class DataLoader:
    def __init__(self, manager: AlphaZeroManager):
        self.manager = manager
        self.self_play_metadata = SelfPlayMetadata(manager.self_play_dir)
        self.n_total = self.self_play_metadata.n_total_positions
        self.n_window = compute_n_window(self.n_total)
        self.window = self.self_play_metadata.get_window(self.n_window)

        self._returned_snapshots = 0
        self._index = len(self.window)

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

        return (input_data, value_data, policy_data)

    def _add_to_minibatch(self, minibatch: List[SelfPlayGameMetadata]):
        if self._index == len(self.window):
            random.shuffle(self.window)
            self._index = 0

        game_metadata = self.window[self._index]
        minibatch.append(game_metadata)
        self._index += 1
        return game_metadata.num_positions


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


def test_vs_perfect(candidate_filename):
    mcts_vs_perfect_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_mcts_vs_perfect')
    n_games = GatingArgs.num_games
    args = [
        mcts_vs_perfect_bin,
        '-G', n_games,
        '-i', GatingArgs.mcts_iters,
        '--nnet-filename', candidate_filename,
    ]
    cmd = ' '.join(map(str, args))
    timed_print(f'Running: {cmd}')
    subprocess_util.run(cmd)

    win_rate = extract_win_score(stdout, 0) / n_games
    print('Perf against perfect: %.5f' % win_rate)


def gating_test(candidate_filename, latest_filename):
    self_play_bin = os.path.join(Repo.root(), 'target/Release/bin/c4_competitive_self_play')
    n_games = GatingArgs.num_games
    args = [
        self_play_bin,
        '-G', n_games,
        '-i', GatingArgs.mcts_iters,
        '-t', GatingArgs.temperature,
        '--nnet-filename', latest_filename,
        '--nnet-filename2', candidate_filename,
    ]
    cmd = ' '.join(map(str, args))
    timed_print(f'Running: {cmd}')
    proc = subprocess_util.Popen(cmd)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        print(stderr)
        raise Exception()

    win_rate = extract_win_score(stdout, 1) / n_games
    promote = win_rate > GatingArgs.promotion_win_rate
    timed_print('Run complete.')
    print(f'Candidate win-rate: %.5f' % win_rate)
    print(f'Promotion win-rate: %.5f' % GatingArgs.promotion_win_rate)
    print(f'Promote: %s' % promote)
    return promote


def main():
    load_args()
    manager = AlphaZeroManager(Args.c4_base_dir)
    manager.write_pid_file()
    latest_model_filename = manager.get_latest_model_filename()

    candidate_filename = manager.get_current_candidate_model_filename()
    checkpoint_filename = manager.get_current_checkpoint_filename()
    if os.path.isfile(checkpoint_filename):
        timed_print(f'Loading checkpoint: {checkpoint_filename}')
        # cp the checkpoint to somewhere local first
        with tempfile.TemporaryDirectory() as tmp:
            tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
            shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
            net = C4Net.load_checkpoint(tmp_checkpoint_filename)
    else:
        # TODO: remove this hard-coded shape
        input_shape = (2, 7, 6)
        net = C4Net(input_shape)
    net.cuda()
    net.train()

    value_loss_lambda = ModelingArgs.value_loss_lambda
    learning_rate = ModelingArgs.learning_rate
    momentum = ModelingArgs.momentum
    weight_decay = ModelingArgs.weight_decay
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    policy_criterion = nn.MultiLabelSoftMarginLoss()
    value_criterion = nn.CrossEntropyLoss()

    epoch = 0
    while True:
        print('******************************')
        epoch += 1
        timed_print(f'Epoch: {epoch}')
        loader = DataLoader(manager)
        timed_print(f'Sampling from the {loader.n_window} most recent positions among {loader.n_total} total positions')

        stats = TrainingStats()

        # TODO: more efficient data loading via torch DataLoader
        for i, data in enumerate(loader):
            inputs, value_labels, policy_labels = data
            inputs = inputs.to('cuda')
            value_labels = value_labels.to('cuda')
            policy_labels = policy_labels.to('cuda')

            optimizer.zero_grad()
            policy_outputs, value_outputs = net(inputs)
            policy_loss = policy_criterion(policy_outputs, policy_labels)
            value_loss = value_criterion(value_outputs, value_labels)
            loss = policy_loss + value_loss * value_loss_lambda

            stats.update(policy_labels, policy_outputs, policy_loss, value_labels, value_outputs, value_loss)

            loss.backward()
            optimizer.step()

        timed_print(f'Epoch {epoch} complete')
        stats.dump()

        net.save_checkpoint(checkpoint_filename)
        net.save_model(candidate_filename)
        timed_print(f'Checkpoint saved')
        if latest_model_filename is None or gating_test(candidate_filename, latest_model_filename):
            # leave out test against perfect for now, more efficient to do this later in a non-blocking path
            # test_vs_perfect(candidate_filename)
            break


if __name__ == '__main__':
    main()
