from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, OptimizerSpec, ShapeInfoDict
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
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


def mini(shape_info_dict: ShapeInfoDict):
    """
    Minimal size, used to just to produce a model to use for unit-testing.
    """
    input_shape = shape_info_dict['input'].shape
    policy_shape = shape_info_dict['policy'].shape
    value_shape = shape_info_dict['value'].shape
    action_value_shape = shape_info_dict['action_value'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)

    assert value_shape == (3,), value_shape

    c_trunk = 1
    c_mid = 1
    c_policy_hidden = 1
    c_action_value_hidden = 1
    c_value_hidden = 1
    n_value_hidden = 1

    return ModelConfig(
        shape_info_dict=shape_info_dict,

        stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),

        blocks=[],

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
        'mini': mini,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 1)

    training_params = TrainingParams(
        target_sample_rate=16,
        minibatches_per_epoch=64,
    )

    training_options = {
        '--mean-noisy-moves': 2,
    }

    rating_params = RatingParams(
        rating_player_options=RatingPlayerOptions(
            num_search_threads=4,
            num_iterations=100,
        ),
        default_target_elo_gap=DefaultTargetEloGap(
            first_run=100.0,
            benchmark=20.0,
        ),
        eval_error_threshold=5.0,
        n_games_per_benchmark=100,
        n_games_per_evaluation=1000,
    )



TicTacToe = TicTacToeSpec()
