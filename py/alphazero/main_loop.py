#!/usr/bin/env python3

import argparse
import os

from alphazero.manager import AlphaZeroManager
from alphazero.optimization_args import ModelingArgs
from config import Config
import games


class Args:
    alphazero_dir: str
    game: str
    tag: str
    restart_gen: int

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tag = args.tag
        Args.restart_gen = args.restart_gen
        assert Args.game, 'Required option: --game/-g'
        assert Args.tag, 'Required option: --tag/-t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-g', '--game', help='game to play (e.g. "c4")')
    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('--restart-gen', type=int, help='gen to resume at')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    ModelingArgs.add_args(parser)

    args = parser.parse_args()
    Args.load(args)
    ModelingArgs.load(args)


def main():
    load_args()
    base_dir = os.path.join(Args.alphazero_dir, Args.game, Args.tag)
    game_type = games.get_game_type(Args.game)
    manager = AlphaZeroManager(game_type, base_dir)
    if Args.restart_gen:
        manager.erase_data_after(Args.restart_gen)
    manager.run(async_mode=not ModelingArgs.synchronous_mode)


if __name__ == '__main__':
    main()
