from typing import Type, Optional

from connect4.tensorizor import C4Net
from neural_net import NeuralNet


class GameType:
    def __init__(self, name: str, net_type: Type[NeuralNet], binary_name: Optional[str] = None):
        self.name = name
        self.net_type = net_type
        self.binary_name = binary_name if binary_name is not None else name


Connect4 = GameType('c4', C4Net)
Othello = GameType('othello', None)  # OthelloNet)


ALL_GAME_TYPES = [Connect4, Othello]
GAME_TYPES_BY_NAME = {game.name: game for game in ALL_GAME_TYPES}
assert len(ALL_GAME_TYPES) == len(GAME_TYPES_BY_NAME)


def get_game_type(game_name) -> GameType:
    if game_name not in GAME_TYPES_BY_NAME:
        raise ValueError(f'Unknown game: {game_name}')
    return GAME_TYPES_BY_NAME[game_name]
