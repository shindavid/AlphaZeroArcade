from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, ShapeInfoDict


BOARD_LENGTH = 3
NUM_ACTIONS = BOARD_LENGTH * BOARD_LENGTH
NUM_PLAYERS = 2
NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES = NUM_PLAYERS + 1  # +1 for empty square


def b7_c32(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    policy_shape = (NUM_ACTIONS, )
    c_trunk = 32
    c_mid = 32
    c_policy_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256

    return ModelConfig(
        shape_info_dict=shape_info_dict,

        stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),

        blocks=[
            ModuleSpec(type='ResBlock', args=['block1', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block2', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block3', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block4', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block5', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block6', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block7', c_trunk, c_mid]),
            ],

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
            ModuleSpec(type='ValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             NUM_PLAYERS]),
            ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            },
        )


@dataclass
class TicTacToeSpec(GameSpec):
    name = 'tictactoe'
    model_configs = {
        'default': b7_c32,
        'b7_c32': b7_c32,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 1)

    # Tic-tac-toe is so simple that most nn evals end up hitting the cache. As a result, the
    # binary tends to be CPU-bound, rather than GPU-bound. Using the default parallelism of 256
    # leads to CPU-starving, so we drop to 16.
    training_options = {
        '-p': 16,
    }

    training_player_options = {
        '-r': 1,
    }

    rating_options = {
        '-p': 16,  # see above comment
    }

    rating_player_options = {
        '-i': 1,
    }


TicTacToe = TicTacToeSpec()
