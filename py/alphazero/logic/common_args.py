from config import Config
import games
from util.py_util import is_valid_path_component

import argparse
from typing import List


class CommonArgs:
    """
    Args used by all alphazero servers.
    """
    alphazero_dir: str
    game: str
    tag: str

    @staticmethod
    def load(args):
        CommonArgs.alphazero_dir = args.alphazero_dir
        CommonArgs.game = args.game
        CommonArgs.tag = args.tag
        CommonArgs.validate()

    @staticmethod
    def validate():
        assert CommonArgs.game, 'Required option: --game/-g'
        assert CommonArgs.tag, 'Required option: --tag/-t'

        assert games.is_valid_game_name(CommonArgs.game), f'Invalid game name: {CommonArgs.game}'
        assert CommonArgs.tag.find('@') == -1, 'Tag cannot contain @'
        assert is_valid_path_component(CommonArgs.tag), f'Illegal tag name: {CommonArgs.tag}'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Run options')

        cfg = Config.instance()

        games.add_parser_argument(group, '-g', '--game')
        group.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
        cfg.add_parser_argument('alphazero_dir', group,
                                '--alphazero-dir', help=argparse.SUPPRESS)

    @staticmethod
    def add_to_cmd(cmd: List[str]):
        cmd.extend([
            '--alphazero-dir', CommonArgs.alphazero_dir,
            '--game', CommonArgs.game,
            '--tag', CommonArgs.tag,
        ])
