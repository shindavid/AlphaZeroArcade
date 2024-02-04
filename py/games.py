import argparse
from typing import Dict, Optional

from net_modules import ModelConfigGenerator
from connect4.connect4_model_configs import C4_MODEL_CONFIGS
from othello.othello_model_configs import OTHELLO_MODEL_CONFIGS
from tictactoe.tictactoe_model_configs import TICTACTOE_MODEL_CONFIGS


class ReferencePlayerFamily:
    def __init__(self, type_str: str, strength_param: str, min_strength, max_strength):
        self.type_str = type_str
        self.strength_param = strength_param
        self.min_strength = min_strength
        self.max_strength = max_strength


class GameType:
    def __init__(self, name: str, model_dict: Dict[str, ModelConfigGenerator],
                 reference_player_family: Optional[ReferencePlayerFamily] = None,
                 binary_name: Optional[str] = None):
        self.name = name
        self.model_dict = model_dict
        self.reference_player_family = reference_player_family
        self.binary_name = binary_name if binary_name is not None else name


Connect4 = GameType('c4', C4_MODEL_CONFIGS, ReferencePlayerFamily('Perfect', '--strength', 0, 21))
Othello = GameType('othello', OTHELLO_MODEL_CONFIGS, ReferencePlayerFamily('edax', '--depth', 0, 21))
TicTacToe = GameType('tictactoe', TICTACTOE_MODEL_CONFIGS, ReferencePlayerFamily('Perfect', '--strength', 0, 1))


ALL_GAME_TYPES = [Connect4, Othello, TicTacToe]
GAME_TYPES_BY_NAME = {game.name: game for game in ALL_GAME_TYPES}
assert len(ALL_GAME_TYPES) == len(GAME_TYPES_BY_NAME)


def get_game_type(game_name) -> GameType:
    if game_name not in GAME_TYPES_BY_NAME:
        raise ValueError(f'Unknown game: {game_name}')
    return GAME_TYPES_BY_NAME[game_name]


def is_valid_game_name(game_name) -> bool:
    return game_name in GAME_TYPES_BY_NAME


def add_parser_argument(parser: argparse.ArgumentParser, *args):
    choices = [t.name for t in ALL_GAME_TYPES]
    parser.add_argument(*args, choices=choices, help='game to play')
