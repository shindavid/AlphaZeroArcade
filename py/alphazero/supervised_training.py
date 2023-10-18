#!/usr/bin/env python3
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
from alphazero.manager import NetTrainer


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
    parser.add_argument('-e', '--epochs', type=int, default='1', help='the number of epochs')
    parser.add_argument('-g', '--game', help='the game')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    ModelingArgs.add_args(parser)

    args = parser.parse_args()
    Args.load(args)
    ModelingArgs.load(args)


def main():
    load_args()
    game = Args.game
    game_type = games.get_game_type(game)

    base_dir = os.path.join(Args.alphazero_dir, game, Args.tag)
    self_play_data_dir = os.path.join(base_dir, 'self-play-data')

    dataset = GamesDataset(self_play_data_dir)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ModelingArgs.minibatch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

    target_names = loader.dataset.get_target_names()

    input_shape = loader.dataset.get_input_shape()
    net = game_type.net_type.create(input_shape, target_names)
    net.cuda('cuda')
    net.train()

    learning_rate = ModelingArgs.learning_rate
    momentum = ModelingArgs.momentum
    weight_decay = ModelingArgs.weight_decay
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    trainer = NetTrainer(ModelingArgs.snapshot_steps)
    for epoch in range(Args.epochs):
        trainer.reset()
        print(f'Epoch: {epoch}/{Args.epochs}')
        trainer.do_training_epoch(loader, net, optimizer, dataset)


if __name__ == '__main__':
    main()
