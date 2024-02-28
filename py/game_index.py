import argparse

from alphazero.game_spec import GameSpec
from games.connect4.spec import Connect4
from games.othello.spec import Othello
from games.tictactoe.spec import TicTacToe


ALL_GAME_SPECS = [
    Connect4,
    Othello,
    TicTacToe,
]

GAME_SPECS_BY_NAME = {game.name: game for game in ALL_GAME_SPECS}
assert len(ALL_GAME_SPECS) == len(GAME_SPECS_BY_NAME)


def get_game_spec(game_name) -> GameSpec:
    if game_name not in GAME_SPECS_BY_NAME:
        raise ValueError(f'Unknown game: {game_name}')
    return GAME_SPECS_BY_NAME[game_name]


def is_valid_game_name(game_name) -> bool:
    return game_name in GAME_SPECS_BY_NAME


def add_parser_argument(parser: argparse.ArgumentParser, *args):
    choices = [t.name for t in ALL_GAME_SPECS]
    parser.add_argument(*args, choices=choices, help='game to play')
