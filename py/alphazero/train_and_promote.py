#!/usr/bin/env python3

import argparse

from alphazero import shared
from config import Config


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
    shared.add_optimization_args(parser)

    args = parser.parse_args()
    Args.load(args)
    shared.OptimizationArgs.load(args)


def main():
    load_args()
    manager = shared.AlphaZeroManager(Args.c4_base_dir)
    raise Exception('TODO')


if __name__ == '__main__':
    main()
