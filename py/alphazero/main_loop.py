#!/usr/bin/env python3

import argparse
import os

from alphazero.manager import AlphaZeroManager
from alphazero.optimization_args import ModelingArgs
from config import Config


# torch.set_default_dtype(torch.float64)


class Args:
    c4_base_dir_root: str
    tag: str
    restart_gen: int

    @staticmethod
    def load(args):
        Args.c4_base_dir_root = args.c4_base_dir_root
        Args.tag = args.tag
        Args.restart_gen = args.restart_gen
        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('--restart-gen', type=int, help='gen to resume at')
    cfg.add_parser_argument('c4.base_dir_root', parser, '-d', '--c4-base-dir-root', help='base-dir-root for game/model files')
    ModelingArgs.add_args(parser)

    args = parser.parse_args()
    Args.load(args)
    ModelingArgs.load(args)


def main():
    load_args()
    c4_base_dir = os.path.join(Args.c4_base_dir_root, Args.tag)
    manager = AlphaZeroManager(c4_base_dir)
    if Args.restart_gen:
        manager.erase_data_after(Args.restart_gen)
    manager.run(async_mode=not ModelingArgs.synchronous_mode)


if __name__ == '__main__':
    main()
