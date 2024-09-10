from dataclasses import dataclass
import math

from games.game_spec import GameSpec
from shared.net_modules import ModelConfig, ModuleSpec, ShapeInfoDict
from shared.training_params import TrainingParams

# constants copied from c++:
class CppConstants:
    kNumColors = 4
    kNumPlayers = kNumColors
    kMaxScore = 63
    kBoardDimension = 20
    kNumCells = kBoardDimension * kBoardDimension
    kNumPieceOrientationCorners = 309
    kNumActions = kNumCells + kNumPieceOrientationCorners + 1


NUM_PLAYERS = CppConstants.kNumPlayers
POLICY_SIZE = CppConstants.kNumActions
NUM_POSSIBLE_SCORES = CppConstants.kMaxScore + 1


def b20_c64(shape_info_dict: ShapeInfoDict):
    input_shape = shape_info_dict['input'].shape
    ownership_shape = shape_info_dict['ownership'].shape
    score_shape = shape_info_dict['score'].shape
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    c_trunk = 64
    c_mid = 64
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
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
            ModuleSpec(type='ResBlock', args=['block20', c_trunk, c_mid]),
        ],

        # TODO: bring back score heads once we fix up ScoreHead to work with dim>2 score-shapes.

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, POLICY_SIZE]),
            ModuleSpec(type='ValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             NUM_PLAYERS]),
            ModuleSpec(type='ActionValueHead',
                       args=['action-value', board_size, c_trunk, c_policy_hidden, POLICY_SIZE]),
            ModuleSpec(type='ScoreHead',
                       args=['score', c_trunk, c_score_margin_hidden,
                             n_score_margin_hidden, score_shape]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden, ownership_shape]),
            ModuleSpec(type='ScoreHead',
                       args=['dummy-score', c_trunk, c_score_margin_hidden,
                             n_score_margin_hidden, score_shape]),
            ModuleSpec(type='OwnershipHead',
                       args=['dummy-ownership', c_trunk, c_ownership_hidden, ownership_shape]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'action-value': 1.0,
            'score': 0.02,
            'ownership': 0.15,
            'dummy-score': 0.02,
            'dummy-ownership': 0.15,
        },
    )


@dataclass
class BlokusSpec(GameSpec):
    name = 'blokus'
    num_players = NUM_PLAYERS
    model_configs = {
        'default': b20_c64,
        'b20_c64': b20_c64,
    }

    training_params = TrainingParams(
        minibatches_per_epoch=256,
        minibatch_size=64,
    )

    training_player_options = {
        '-r': 16,
    }


Blokus = BlokusSpec()
