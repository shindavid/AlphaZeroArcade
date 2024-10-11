from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModuleSpec, ShapeInfoDict
from shared.training_params import TrainingParams


def b19_c128(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    policy_shape = shape_info_dict['policy'].shape
    value_shape = shape_info_dict['value'].shape
    action_value_shape = shape_info_dict['action_value'].shape
    ownership_shape = shape_info_dict['ownership'].shape
    score_margin_shape = shape_info_dict['score_margin'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)

    assert value_shape == (3,), value_shape

    c_trunk = 128
    c_mid = 128
    c_gpool = 32
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_action_value_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256
    c_score_margin_hidden = 32
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
            ModuleSpec(type='ResBlockWithGlobalPooling', args=['block5', c_trunk, c_mid, c_gpool]),

            ModuleSpec(type='ResBlock', args=['block6', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block7', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block8', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block9', c_trunk, c_mid]),
            ModuleSpec(type='ResBlockWithGlobalPooling', args=['block10', c_trunk, c_mid, c_gpool]),

            ModuleSpec(type='ResBlock', args=['block11', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block12', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block13', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block14', c_trunk, c_mid]),
            ModuleSpec(type='ResBlockWithGlobalPooling', args=['block15', c_trunk, c_mid, c_gpool]),

            ModuleSpec(type='ResBlock', args=['block16', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block17', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block18', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block19', c_trunk, c_mid]),
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
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_shape]),
            ModuleSpec(type='ScoreHead',
                       args=['score_margin', c_trunk, c_score_margin_hidden,
                             n_score_margin_hidden, score_margin_shape]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden, ownership_shape]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'action_value': 2.0,
            'opp_policy': 0.15,
            'score_margin': 0.02,
            'ownership': 0.15,
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
        'default': b19_c128,
        'b19_c128': b19_c128,
    }
    reference_player_family = ReferencePlayerFamily('edax', '--depth', 0, 21)

    training_params = TrainingParams(
        target_sample_rate=64,
        minibatches_per_epoch=500,
        minibatch_size=100,
    )

    training_player_options = {
        '-r': 4,
    }

    rating_options = {
        '-p': 50,  # edax player hogs too much CPU/memory, so limit parallelism
    }

    rating_player_options = {
        '-i': 400,
        '-n': 8,
    }


Othello = OthelloSpec()
