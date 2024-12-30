from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, OptimizerSpec, ShapeInfoDict
from shared.training_params import TrainingParams


def b7_c128(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    policy_shape = shape_info_dict['policy'].shape
    value_shape = shape_info_dict['value'].shape
    action_value_shape = shape_info_dict['action_value'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)

    assert value_shape == (3,), value_shape

    c_trunk = 128
    c_mid = 128
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_action_value_hidden = 2
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

        neck=None,
        # neck=ModuleSpec(type='KataGoNeck', args=[c_trunk]),

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
            ModuleSpec(type='WinLossDrawValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden]),
            ModuleSpec(type='WinShareActionValueHead',
                       args=['action_value', board_size, c_trunk, c_action_value_hidden,
                             action_value_shape]),
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_shape]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'action_value': 1.0,
            'opp_policy': 0.15,
        },

        opt=OptimizerSpec(type='SGD', kwargs={'lr': 6e-5, 'weight_decay': 6e-5, 'momentum': 0.9}),
    )


def transformer(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    policy_shape = shape_info_dict['policy'].shape
    value_shape = shape_info_dict['value'].shape
    action_value_shape = shape_info_dict['action_value'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)

    assert value_shape == (3,), value_shape

    embed_dim = 64
    n_heads = 8
    n_layers = 8
    c_trunk = 128

    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_action_value_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256

    return ModelConfig(
        shape_info_dict=shape_info_dict,

        stem=ModuleSpec(type='TransformerBlock', args=[
                        input_shape, embed_dim, n_heads, n_layers, c_trunk]),

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
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_shape]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'action_value': 1.0,
            'opp_policy': 0.15,
        },

        opt=OptimizerSpec(type='RAdam', kwargs={'lr': 6e-5, 'weight_decay': 6e-5}),
    )


@dataclass
class Connect4Spec(GameSpec):
    name = 'c4'
    extra_runtime_deps = ['extra_deps/connect4/c4solver',
                          'extra_deps/connect4/7x6.book']
    model_configs = {
        'b7_c128': b7_c128,
        'transformer': transformer,
        'default': transformer,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 21)

    training_params = TrainingParams(
        target_sample_rate=8,
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
