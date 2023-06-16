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
from alphazero.manager import TrainingStats


class Args:
    alphazero_dir: str
    tag: str
    epochs: int
    game: str

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.tag = args.tag
        Args.epochs = args.epochs
        Args.game = args.game
        assert Args.epochs > 0, 'epochs has to be positive'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('--epochs', type=int, default='1', help='the number of epochs')
    parser.add_argument('--game', choices=['othello', 'connect4'], help='the game')
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

    dataset = GamesDataset(self_play_data_dir, first_gen=100)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ModelingArgs.minibatch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

    timed_print(f'Training off of {dataset.n_total_games} games')
    input_shape = loader.dataset.get_input_shape()
    net = game_type.net_type.create(input_shape)
    net.cuda('cuda')
    net.train()

    learning_rate = ModelingArgs.learning_rate
    momentum = ModelingArgs.momentum
    weight_decay = ModelingArgs.weight_decay
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    value_loss_lambda = ModelingArgs.value_loss_lambda
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.CrossEntropyLoss()
    steps = 0
    for epoch in range(Args.epochs):
        print(f'epoch: {epoch}/{Args.epochs}')
        stats = TrainingStats()
        for data in loader:
            inputs, value_labels, policy_labels = data
            inputs = inputs.type(torch.float32).to('cuda')
            value_labels = value_labels.to('cuda')
            policy_labels = policy_labels.to('cuda')

            optimizer.zero_grad()
            policy_outputs, value_outputs = net(inputs)
            n = policy_outputs.shape[0]
            policy_loss = policy_criterion(policy_outputs.reshape((n, -1)), policy_labels.reshape((n, -1)))
            value_loss = value_criterion(value_outputs, value_labels)
            loss = policy_loss + value_loss * value_loss_lambda

            stats.update(policy_labels, policy_outputs, policy_loss, value_labels, value_outputs, value_loss)

            loss.backward()
            optimizer.step()
            steps += 1
            if steps == ModelingArgs.snapshot_steps:
                break
        stats.dump()



if __name__ == '__main__':
    main()
