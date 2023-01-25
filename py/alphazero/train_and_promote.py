#!/usr/bin/env python3

import argparse

from alphazero import shared
from config import Config


class Args:
    window_alpha: float = 0.75
    window_beta: float = 0.4
    window_c: int = 250000
    c4_base_dir: str

    @staticmethod
    def load(args):
        Args.window_alpha = args.window_alpha
        Args.window_beta = args.window_beta
        Args.window_c = args.window_c
        Args.c4_base_dir = args.c4_base_dir

        assert Args.c4_base_dir, 'Required option: -d'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument("-A", "--window-alpha", type=float, default=Args.window_alpha,
                        help='alpha for n_window formula (default: %(default)s)')
    parser.add_argument("-B", "--window-beta", type=float, default=Args.window_beta,
                        help='beta for n_window formula (default: %(default)s)')
    parser.add_argument("-c", "--window-c", type=int, default=Args.window_c,
                        help='beta for n_window formula (default: %(default)s)')

    cfg.add_parser_argument('c4.base_dir', parser, '-d', '--c4-base-dir', help='base-dir for game/model files')
    return parser.parse_args()


def main():
    load_args()
    manager = shared.AlphaZeroManager(Args.c4_base_dir)
    raise Exception('TODO')


if __name__ == '__main__':
    main()
