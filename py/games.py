from typing import Type, Optional

from connect4.connect4_net import C4Net
from neural_net import NeuralNet
from othello.othello_net import OthelloNet


class ReferencePlayerFamily:
    def __init__(self, type_str: str, strength_param: str, min_strength, max_strength):
        self.type_str = type_str
        self.strength_param = strength_param
        self.min_strength = min_strength
        self.max_strength = max_strength


class GameType:
    def __init__(self, name: str, net_type: Type[NeuralNet],
                 reference_player_family: Optional[ReferencePlayerFamily] = None,
                 binary_name: Optional[str] = None):
        self.name = name
        self.net_type = net_type
        self.reference_player_family = reference_player_family
        self.binary_name = binary_name if binary_name is not None else name


Connect4 = GameType('c4', C4Net, ReferencePlayerFamily('Perfect', '--strength', 0, 21))
Othello = GameType('othello', OthelloNet, ReferencePlayerFamily('edax', '--depth', 0, 21))


ALL_GAME_TYPES = [Connect4, Othello]
GAME_TYPES_BY_NAME = {game.name: game for game in ALL_GAME_TYPES}
assert len(ALL_GAME_TYPES) == len(GAME_TYPES_BY_NAME)


def get_game_type(game_name) -> GameType:
    if game_name not in GAME_TYPES_BY_NAME:
        raise ValueError(f'Unknown game: {game_name}')
    return GAME_TYPES_BY_NAME[game_name]
