#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
from torch import optim

import games
from alphazero.data.games_dataset import GamesDataset
from alphazero.net_trainer import NetTrainer
from alphazero.optimization_args import ModelingArgs
from config import Config
from util.py_util import timed_print


class Args:
    alphazero_dir: str
    tag: str
    epochs: int
    game: str
    checkpoint_filename: str
    cuda_device_str: str

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.tag = args.tag
        Args.epochs = args.epochs
        Args.game = args.game
        Args.checkpoint_filename = args.checkpoint_filename
        Args.cuda_device_str = args.cuda_device_str
        assert Args.epochs > 0, 'epochs has to be positive'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('-g', '--game', help='the game')
    parser.add_argument('-C', '--checkpoint-filename', help='checkpoint filename')
    parser.add_argument('-D', '--cuda-device-str', default='cuda:0', help='cuda device str')
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

    if Args.checkpoint_filename and os.path.isfile(Args.checkpoint_filename):
        checkpoint = torch.load(Args.checkpoint_filename)
        net = game_type.net_type.load_from_checkpoint(checkpoint)
        epoch = checkpoint['epoch']
    else:
        target_names = loader.dataset.get_target_names()
        input_shape = loader.dataset.get_input_shape()
        net = game_type.net_type(input_shape, target_names)
        epoch = 0

    net.cuda(Args.cuda_device_str)
    net.train()

    learning_rate = ModelingArgs.learning_rate
    momentum = ModelingArgs.momentum
    weight_decay = ModelingArgs.weight_decay
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    trainer = NetTrainer(ModelingArgs.snapshot_steps, Args.cuda_device_str)
    while epoch < Args.epochs:
        trainer.reset()
        print(f'Epoch: {epoch}/{Args.epochs}')
        trainer.do_training_epoch(loader, net, optimizer, dataset)

        if Args.checkpoint_filename:
            checkpoint = {
                'epoch': epoch,
                }
            net.add_to_checkpoint(checkpoint)
            torch.save(checkpoint, Args.checkpoint_filename)

        epoch += 1

if __name__ == '__main__':
    main()
