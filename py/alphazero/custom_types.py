from config import Config

import argparse
from dataclasses import dataclass

Generation = int


@dataclass
class RunInfo:
    alphazero_dir: str
    game: str
    tag: str

    def __post_init__(self):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        cfg = Config.instance()

        parser.add_argument('-g', '--game', help='the game')
        parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')

        cfg.add_parser_argument('alphazero_dir', parser, '--alphazero-dir', help='alphazero directory')
