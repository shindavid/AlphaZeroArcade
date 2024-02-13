from config import Config
import games
from util.py_util import is_valid_path_component

import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class CommonParams:
    """
    Params used by all alphazero servers.
    """
    alphazero_dir: str
    game: str
    tag: str

    def __post_init__(self):
        self.validate()

    @staticmethod
    def create(args) -> 'CommonParams':
        return CommonParams(
            alphazero_dir=args.alphazero_dir,
            game=args.game,
            tag=args.tag,
        )

    def validate(self):
        assert self.game, 'Required option: --game/-g'
        assert self.tag, 'Required option: --tag/-t'

        assert games.is_valid_game_name(self.game), f'Invalid game name: {self.game}'
        assert self.tag.find('@') == -1, 'Tag cannot contain @'
        assert is_valid_path_component(self.tag), f'Illegal tag name: {self.tag}'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, multiple_tags=False):
        group = parser.add_argument_group('Common options')

        cfg = Config.instance()

        games.add_parser_argument(group, '-g', '--game')
        if multiple_tags:
            group.add_argument('-t', '--tag', help='comma-separated tags for this run (e.g. "v1,v2")')
        else:
            group.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
        cfg.add_parser_argument('alphazero_dir', group,
                                '--alphazero-dir', help=argparse.SUPPRESS)

    def add_to_cmd(self, cmd: List[str]):
        cfg = Config.instance()

        if cfg.get('alphazero_dir') != self.alphazero_dir:
            cmd.append('--alphazero-dir')
            cmd.append(self.alphazero_dir)

        cmd.extend([
            '--game', self.game,
            '--tag', self.tag,
        ])
