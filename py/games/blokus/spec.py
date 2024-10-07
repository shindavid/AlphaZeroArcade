from dataclasses import dataclass
import math

from games.game_spec import GameSpec
from shared.net_modules import ModelConfig, ModuleSpec, ShapeInfoDict
from shared.training_params import TrainingParams


def b20_c128(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    policy_shape = shape_info_dict['policy'].shape
    value_shape = shape_info_dict['value'].shape
    action_value_shape = shape_info_dict['action_value'].shape
    ownership_shape = shape_info_dict['ownership'].shape
    score_shape = shape_info_dict['score'].shape
    unplayed_pieces_shape = shape_info_dict['unplayed_pieces'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)

    c_trunk = 128
    c_mid = 128
    c_gpool = 32
    c_policy_hidden = 2
    c_action_value_hidden = 16
    c_value_hidden = 1
    n_value_hidden = 256
    c_score_margin_hidden = 32
    n_score_margin_hidden = 32
    c_ownership_hidden = 64
    c_unplayed_pieces_hidden = 32
    n_unplayed_pieces_hidden = 32

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
            ModuleSpec(type='ResBlock', args=['block20', c_trunk, c_mid]),
        ],

        neck=None,

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
            ModuleSpec(type='WinShareValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             value_shape]),
            ModuleSpec(type='WinShareActionValueHead',
                       args=['action_value', board_size, c_trunk, c_action_value_hidden,
                             action_value_shape]),
            ModuleSpec(type='ScoreHead',
                       args=['score', c_trunk, c_score_margin_hidden,
                             n_score_margin_hidden, score_shape]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden, ownership_shape]),
            ModuleSpec(type='GeneralLogitHead',
                       args=['unplayed_pieces', c_trunk, c_unplayed_pieces_hidden,
                             n_unplayed_pieces_hidden, unplayed_pieces_shape]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'action_value': 1.0,
            'score': 0.02,
            'ownership': 0.15,
            'unplayed_pieces': 0.3,
        },
    )


@dataclass
class BlokusSpec(GameSpec):
    name = 'blokus'
    num_players = 4
    model_configs = {
        'default': b20_c128,
        'b20_c128': b20_c128,
    }

    training_params = TrainingParams(
        target_sample_rate=32,
        minibatches_per_epoch=256,
        minibatch_size=64,
    )

    training_player_options = {
        '-r': 8,
    }


Blokus = BlokusSpec()
