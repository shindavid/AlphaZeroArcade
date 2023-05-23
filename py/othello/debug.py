#!/usr/bin/env python3

"""
Trains a neural network straight from the self-play training data, and checks for symmetry conditions on the resultant
model on the starting position of the game.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch import optim

import games
from alphazero.data.games_dataset import GamesDataset
from alphazero.optimization_args import ModelingArgs
from config import Config
from util.py_util import timed_print


class Args:
    alphazero_dir: str
    tag: str
    start_gen: int
    omit_passes: bool
    starting_position_only: bool

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.tag = args.tag
        Args.start_gen = args.start_gen
        Args.omit_passes = bool(args.omit_passes)
        Args.starting_position_only = bool(args.starting_position_only)

        assert not Args.omit_passes and Args.starting_position_only, 'Use -o or -O, not both'
        assert Args.tag, 'Required option: --tag/-t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('-g', '--start-gen', type=int, default=5800, help='gen to start at')
    parser.add_argument('-o', '--omit-passes', action='store_true', help='omit passes from training data')
    parser.add_argument('-O', '--starting-position-only', action='store_true', help='use starting position only')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    ModelingArgs.add_args(parser)

    args = parser.parse_args()
    Args.load(args)
    ModelingArgs.load(args)


def main():
    load_args()
    game = 'othello'
    game_type = games.get_game_type(game)

    base_dir = os.path.join(Args.alphazero_dir, game, Args.tag)
    self_play_data_dir = os.path.join(base_dir, 'self-play-data')

    dataset = GamesDataset(self_play_data_dir, first_gen=Args.start_gen)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ModelingArgs.minibatch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

    timed_print(f'Training off of {dataset.n_total_games} games')
    input_shape = loader.dataset.get_input_shape()
    net = game_type.net_type.create(input_shape)
    net.cuda(device=1)
    net.train()

    learning_rate = ModelingArgs.learning_rate
    momentum = ModelingArgs.momentum
    weight_decay = ModelingArgs.weight_decay
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # value_loss_lambda = ModelingArgs.value_loss_lambda
    policy_criterion = nn.CrossEntropyLoss()
    # value_criterion = nn.CrossEntropyLoss()

    starting_position = torch.zeros((1, 2, 8, 8)).to(device=1)
    starting_position[0, 0, 3, 4] = 1
    starting_position[0, 0, 4, 3] = 1
    starting_position[0, 1, 3, 3] = 1
    starting_position[0, 1, 4, 4] = 1

    num_rows = 0
    batch_num = 0
    filtered_rows = 0

    policy_sum = torch.zeros((8, 8)).to(device=1)
    policy_count = 0

    for epoch in range(5):
        for data in loader:
            inputs, value_labels, policy_labels = data
            inputs = inputs.type(torch.float32).to(device=1)
            # value_labels = value_labels.to(device=1)
            policy_labels = policy_labels.to(device=1)

            if Args.omit_passes:
                n_original_rows = inputs.shape[0]
                non_pass_indices = torch.where(policy_labels[:, 3, 3] < 0.5)[0]
                inputs = inputs[non_pass_indices]
                policy_labels = policy_labels[non_pass_indices]
                value_labels = value_labels[non_pass_indices]
                filtered_rows += n_original_rows - inputs.shape[0]
            elif Args.starting_position_only:
                n_original_rows = inputs.shape[0]
                starting_filter = (torch.sum(inputs, dim=(1, 2, 3)) == 4) & (inputs[:, 0, 3, 4] == 1)
                starting_position_indices = torch.where(starting_filter)[0]
                inputs = inputs[starting_position_indices]
                policy_labels = policy_labels[starting_position_indices]

                policy_sum += torch.sum(policy_labels, dim=0)
                policy_count += policy_labels.shape[0]
                filtered_rows += n_original_rows - inputs.shape[0]

            optimizer.zero_grad()
            policy_outputs, value_outputs = net(inputs)
            n = policy_outputs.shape[0]
            policy_loss = policy_criterion(policy_outputs.reshape((n, -1)), policy_labels.reshape((n, -1)))
            # value_loss = value_criterion(value_outputs, value_labels)
            # loss = policy_loss + value_loss * value_loss_lambda
            loss = policy_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                net.eval()
                logit_policy = net(starting_position)[0][0]
                policy = torch.softmax(logit_policy.flatten(), dim=0).reshape((8, 8))
                subpolicy = [policy[2, 3], policy[3, 2], policy[4, 5], policy[5, 4]]
                spread = max(subpolicy) - min(subpolicy)
                mass = sum(subpolicy)
                net.train()

            num_rows += inputs.shape[0]
            batch_num += 1
            if batch_num % 100 == 0:
                timed_print('Epoch %d Processed %8d rows, filtered %8d rows, spread=%.3f sub_mass=%.3f subpolicy=[%.3f %.3f %.3f %.3f]' %
                            (epoch+1, num_rows, filtered_rows, spread, mass, subpolicy[0], subpolicy[1], subpolicy[2], subpolicy[3]))

                if Args.starting_position_only:
                    policy_avg = policy_sum / max(1, policy_count)
                    print(policy_avg)


if __name__ == '__main__':
    main()
