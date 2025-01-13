from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, OptimizerSpec, ShapeInfoDict
from shared.training_params import TrainingParams


def b3_c32(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    policy_shape = shape_info_dict['policy'].shape
    value_shape = shape_info_dict['value'].shape
    action_value_shape = shape_info_dict['action_value'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)

    assert value_shape == (3,), value_shape

    c_trunk = 32
    c_mid = 32
    c_policy_hidden = 2
    c_action_value_hidden = 4
    c_value_hidden = 2
    n_value_hidden = 32

    return ModelConfig(
        shape_info_dict=shape_info_dict,

        stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),

        blocks=[
            ModuleSpec(type='ResBlock', args=['block1', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block2', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block3', c_trunk, c_mid]),
            ],

        neck=None,

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
            ModuleSpec(type='WinLossDrawValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden]),
            ModuleSpec(type='WinShareActionValueHead',
                       args=['action_value', board_size, c_trunk, c_action_value_hidden,
                             action_value_shape]),
            ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'action_value': 1.0,
        },

        opt=OptimizerSpec(type='RAdam', kwargs={'lr': 6e-5, 'weight_decay': 6e-5}),
    )


@dataclass
class TicTacToeSpec(GameSpec):
    name = 'tictactoe'
    model_configs = {
        'default': b3_c32,
        'b3_c32': b3_c32,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 1)

    training_params = TrainingParams(
        target_sample_rate=16,
        minibatches_per_epoch=64,
    )

    # Tic-tac-toe is so simple that most nn evals end up hitting the cache. As a result, the
    # binary tends to be CPU-bound, rather than GPU-bound. Using the default parallelism of 256
    # leads to CPU-starving, so we drop it.
    training_options = {
        '-p': 16,
        '--mean-noisy-moves': 2,
    }

    rating_options = {
        '-p': 64,  # see above comment
    }

    rating_player_options = {
        '-i': 100,
        '--starting-move-temp': 0,  # zero-move-temp so we don't do silly misplays
        '--ending-move-temp': 0,    # zero-move-temp so we don't do silly misplays
    }


TicTacToe = TicTacToeSpec()
