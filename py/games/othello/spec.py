from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModelConfigGenerator, ModuleSpec, OptimizerSpec, \
    ShapeInfoDict
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams


class CNN_b9_c128(ModelConfigGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
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

            opt=OptimizerSpec(type='RAdam', kwargs={'lr': 6e-5, 'weight_decay': 6e-5}),
        )


def chessformer(shape_info_dict: ShapeInfoDict):
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

        stem=ModuleSpec(type='ChessformerBlock', args=[
                        input_shape, embed_dim, n_heads, n_layers, c_trunk],
                        kwargs={
                            'use_static_bias': True,    # learned T×T per-head bias
                            'use_shaw': True,           # pairwise aQ/aK/aV
                            'use_smolgen': True,        # dynamic T×T logits (shared 256→T^2)
                            'ffn_multiplier': 1.0       # small FFN ≈ embed_dim),
                        }),

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

        opt=OptimizerSpec(type='RAdam', kwargs={'lr': 5e-4, 'weight_decay': 6e-5}),
    )


@dataclass
class OthelloSpec(GameSpec):
    name = 'othello'
    extra_runtime_deps = [
        'extra_deps/edax-reversi/bin/lEdax-x64-modern',
        'extra_deps/edax-reversi/data/book.dat',
        'extra_deps/edax-reversi/data/eval.dat',
        ]
    model_configs = {
        'default': CNN_b9_c128,
        'b9_c128': CNN_b9_c128,
    }
    reference_player_family = ReferencePlayerFamily('edax', '--depth', 0, 15)
    ref_neighborhood_size = 5

    training_params = TrainingParams(
        target_sample_rate=32,
        minibatches_per_epoch=500,
        minibatch_size=100,
    )

    training_options = {
        '--mean-noisy-moves': 4,
    }

    rating_params = RatingParams(
        rating_player_options=RatingPlayerOptions(
            num_search_threads=4,
            num_iterations=100,
        ),
        default_target_elo_gap=DefaultTargetEloGap(
            first_run=500.0,
            benchmark=100.0,
        ),
        eval_error_threshold=50.0,
        n_games_per_self_evaluation=100,
        n_games_per_evaluation=300,
    )


Othello = OthelloSpec()
