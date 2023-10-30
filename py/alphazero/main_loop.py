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
    binary_path: str
    tag: str
    model_cfg: str
    fork_from: str
    restart_gen: int
    synchronous_mode: bool

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.binary_path = args.binary_path
        Args.tag = args.tag
        Args.model_cfg = args.model_cfg
        Args.fork_from = args.fork_from
        Args.restart_gen = args.restart_gen
        Args.synchronous_mode = args.synchronous_mode
        assert Args.game, 'Required option: --game/-g'
        assert Args.tag, 'Required option: --tag/-t'
        assert Args.tag.find('@') == -1, 'Tag cannot contain @'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-g', '--game', help='game to play (e.g. "c4")')
    parser.add_argument('-b', '--binary-path',
                        help='binary path. By default, if a unique binary is found in the '
                        'alphazero dir, it will be used. If no binary is found in the alphazero '
                        'dir, then will use one found in REPO_ROOT/target/Release/bin/. If '
                        'multiple binaries are found in the alphazero dir, then this option is '
                        'required.')
    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('-m', '--model-cfg', default='default', help='model config (default: %(default)s)')
    parser.add_argument('-f', '--fork-from', help='tag to fork off of (e.g., "v1", or "v1@100" to fork off of gen 100))')
    parser.add_argument('--restart-gen', type=int, help='gen to resume at')
    parser.add_argument('-S', '--synchronous-mode', action='store_true',
                        help='synchronous mode (default: asynchronous)')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    ModelingArgs.add_args(parser)

    args = parser.parse_args()
    Args.load(args)
    ModelingArgs.load(args)


def main():
    load_args()
    base_dir = os.path.join(Args.alphazero_dir, Args.game, Args.tag)
    game_type = games.get_game_type(Args.game)
    manager = AlphaZeroManager(game_type, base_dir, Args.binary_path)
    manager.makedirs()
    if Args.fork_from:
        fork_from = Args.fork_from
        gen = None
        if '@' in fork_from:
            fork_from, gen = fork_from.split('@')
            gen = int(gen)
        fork_base_dir = os.path.join(Args.alphazero_dir, Args.game, fork_from)
        fork_manager = AlphaZeroManager(game_type, fork_base_dir)
        manager.fork_from(fork_manager, gen)
    if Args.restart_gen:
        manager.erase_data_after(Args.restart_gen)

    manager.set_model_cfg(Args.model_cfg)
    manager.run(async_mode=not Args.synchronous_mode)


if __name__ == '__main__':
    main()
