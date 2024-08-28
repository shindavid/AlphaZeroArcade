from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, ShapeInfoDict
from shared.training_params import TrainingParams

BOARD_LENGTH = 8
NUM_SQUARES = BOARD_LENGTH * BOARD_LENGTH
NUM_PLAYERS = 2
NUM_POSSIBLE_SCORE_MARGINS = 2 * NUM_SQUARES + 1


def b19_c64(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    ownership_shape = shape_info_dict['ownership'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    policy_size = NUM_SQUARES + 1  # + 1 for pass
    c_trunk = 64
    c_mid = 64
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256
    c_score_margin_hidden = 1
    n_score_margin_hidden = 32
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
            ModuleSpec(type='ResBlock', args=['block8', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block9', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block10', c_trunk, c_mid]),

            ModuleSpec(type='ResBlock', args=['block11', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block12', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block13', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block14', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block15', c_trunk, c_mid]),

            ModuleSpec(type='ResBlock', args=['block16', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block17', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block18', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block19', c_trunk, c_mid]),
        ],

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_size]),
            ModuleSpec(type='ValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             NUM_PLAYERS]),
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_size]),
            ModuleSpec(type='ScoreHead',
                       args=['score_margin', board_size, c_trunk, c_score_margin_hidden,
                             n_score_margin_hidden, NUM_POSSIBLE_SCORE_MARGINS]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden, ownership_shape]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'opp_policy': 0.15,
            'score_margin': 0.02,
            'ownership': 0.15
        },
    )


@dataclass
class OthelloSpec(GameSpec):
    name = 'othello'
    extra_runtime_deps = [
        'extra_deps/edax-reversi/bin/lEdax-x64-modern',
        'extra_deps/edax-reversi/data',
        ]
    model_configs = {
        'default': b19_c64,
        'b19_c64': b19_c64,
    }
    reference_player_family = ReferencePlayerFamily('edax', '--depth', 0, 21)

    training_params = TrainingParams(
        window_size_function_str='fixed(300000)',
        minibatches_per_epoch=500,
        minibatch_size=100,
    )

    training_player_options = {
        '-r': 4,
    }

    rating_player_options = {
        '-i': 400,
    }


Othello = OthelloSpec()
