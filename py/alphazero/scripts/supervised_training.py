#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
from torch import optim

import game_index
from net_modules import Model
from alphazero.logic.position_dataset import GamesDataset
from alphazero.logic.net_trainer import NetTrainer
from alphazero.logic.training_params import TrainingParams


class Args:
    alphazero_dir: str
    game: str
    tag: str
    model_cfg: str
    epochs: int
    optimizer: str
    checkpoint_filename: str
    cuda_device_str: str

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tag = args.tag
        Args.model_cfg = args.model_cfg
        Args.epochs = args.epochs
        Args.optimizer = args.optimizer
        Args.checkpoint_filename = args.checkpoint_filename
        Args.cuda_device_str = args.cuda_device_str
        assert Args.epochs > 0, 'epochs has to be positive'


def load_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-f', '--test-fraction', type=float, default=0.1,
    #                     help='what fraction of the data to use for testing (default: %(default).2f)')
    parser.add_argument('-g', '--game', help='the game')
    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('-m', '--model-cfg', default='default', help='model config (default: %(default)s)')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('-O', '--optimizer', choices=['SGD', 'Adam'], default='SGD', help='optimizer type')
    parser.add_argument('-C', '--checkpoint-filename', help='checkpoint filename')
    parser.add_argument('-D', '--cuda-device-str', default='cuda:0', help='cuda device str')

    # TODO: CommonParams
    TrainingParams.add_args(parser)

    args = parser.parse_args()
    Args.load(args)
    TrainingParams.load(args)


def main():
    load_args()
    game = Args.game
    game_spec = game_index.get_game_spec(game)

    base_dir = os.path.join(Args.alphazero_dir, game, Args.tag)
    self_play_data_dir = os.path.join(base_dir, 'self-play-data')

    dataset = GamesDataset(self_play_data_dir)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TrainingParams.minibatch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

    checkpoint = {}
    if Args.checkpoint_filename and os.path.isfile(Args.checkpoint_filename):
        checkpoint = torch.load(Args.checkpoint_filename)

    if checkpoint:
        net = Model.load_from_checkpoint(checkpoint)
        epoch = checkpoint['epoch']
    else:
        target_names = loader.dataset.get_target_names()
        input_shape = loader.dataset.get_input_shape()
        net = Model(game_spec.model_configs[Args.model_cfg](input_shape))
        net.validate_targets(target_names)
        epoch = 0

    net.cuda(Args.cuda_device_str)
    net.train()

    learning_rate = TrainingParams.learning_rate
    weight_decay = TrainingParams.weight_decay
    if Args.optimizer == 'SGD':
        momentum = TrainingParams.momentum
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif Args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception(f'Unknown optimizer: {Args.optimizer}')

    if checkpoint and 'opt.state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['opt.state_dict'])

    trainer = NetTrainer(
        TrainingParams.minibatches_per_epoch, Args.cuda_device_str)
    n_samples_processed = 0
    while epoch < Args.epochs:
        trainer.reset()
        print(f'Epoch: {epoch}/{Args.epochs}')
        stats = trainer.do_training_epoch(loader, net, optimizer, dataset)
        stats.dump()
        n_samples_processed += stats.n_samples
        avg_sample_usage = n_samples_processed / dataset.n_window
        trainer.dump_timing_stats()
        print('Average sample usage: %.3f' % avg_sample_usage)
        print('')

        if Args.checkpoint_filename:
            checkpoint = {
                'epoch': epoch,
                'opt.state_dict': optimizer.state_dict(),
                }
            net.add_to_checkpoint(checkpoint)
            torch.save(checkpoint, Args.checkpoint_filename)

        epoch += 1

if __name__ == '__main__':
    main()
