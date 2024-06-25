from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, ShapeInfoDict
from shared.training_params import TrainingParams

from typing import Dict


NUM_COLUMNS = 7
NUM_ROWS = 6
NUM_COLORS = 2
NUM_PLAYERS = 2
NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES = NUM_PLAYERS + 1  # +1 for empty square


def b7_c64(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    policy_size = NUM_COLUMNS
    c_trunk = 64
    c_mid = 64
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256
    c_ownership_hidden = 64

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
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_size]),
            ModuleSpec(type='ValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             NUM_PLAYERS]),
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_size]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden,
                             NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'opp_policy': 0.15,
            'ownership': 0.15
        },
    )


@dataclass
class Connect4Spec(GameSpec):
    name = 'c4'
    extra_runtime_deps = ['extra_deps/connect4/c4solver',
                          'extra_deps/connect4/7x6.book']
    model_configs = {
        'default': b7_c64,
        'b7_c64': b7_c64,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 21)

    training_params = TrainingParams(
        window_size_function_str='fixed(50000)',
        minibatches_per_epoch=500,
        minibatch_size=100,
    )

    training_player_options = {
        '-r': 2,
    }

    rating_player_options = {
        '-i': 100,
        '-n': 4,
    }


Connect4 = Connect4Spec()
