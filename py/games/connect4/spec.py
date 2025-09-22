from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.net_modules import ModelConfig, ModelGenerator, ModuleSpec, OptimizerSpec, \
    SearchParadigm, ShapeInfoDict
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams
from shared.transformer_modules import TransformerBlockParams


class CNN_b7_c128(ModelGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
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
                        args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden,
                              policy_shape]),
            ],

            loss_weights={
                'policy': 1.0,
                'value': 1.5,
                'action_value': 1.0,
                'opp_policy': 0.15,
            },

            opt=OptimizerSpec(type='RAdam', kwargs={'lr': 6e-5, 'weight_decay': 6e-5}),
        )


class CNN_b7_c128_beta0(ModelGenerator):
    search_paradigm: SearchParadigm = SearchParadigm.BetaZero

    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
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
        c_value_uncertainty_hidden = 1
        n_value_uncertainty_hidden = 256

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

            heads=[
                # primary heads
                ModuleSpec(type='PolicyHead',
                        args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
                ModuleSpec(type='WinLossDrawValueHead',
                        args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden]),
                ModuleSpec(type='WinShareActionValueHead',
                        args=['action_value', board_size, c_trunk, c_action_value_hidden,
                              action_value_shape]),
                ModuleSpec(type='GeneralLogitHead',
                        args=['value_uncertainty', c_trunk, c_value_uncertainty_hidden,
                              n_value_uncertainty_hidden, (1, )]),
                ModuleSpec(type='GeneralLogitHead',
                        args=['action_value_uncertainty', c_trunk, c_value_uncertainty_hidden,
                              n_value_uncertainty_hidden, action_value_shape]),

                # auxiliary heads
                ModuleSpec(type='PolicyHead',
                        args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden,
                              policy_shape]),
            ],

            loss_weights={
                # primary heads
                'policy': 1.0,
                'value': 1.5,
                'action_value': 1.0,
                'value_uncertainty': 1.0,
                'action_value_uncertainty': 1.0,

                # auxiliary heads
                'opp_policy': 0.15,
            },

            opt=OptimizerSpec(type='RAdam', kwargs={'lr': 6e-5, 'weight_decay': 6e-5}),
        )


class Transformer(ModelGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
        input_shape = shape_info_dict['input'].shape
        policy_shape = shape_info_dict['policy'].shape
        value_shape = shape_info_dict['value'].shape
        action_value_shape = shape_info_dict['action_value'].shape
        board_shape = input_shape[1:]
        board_size = math.prod(board_shape)

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        cnn_output_shape  = (c_trunk, *board_shape)

        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256

        transformer_block_params = TransformerBlockParams(
            input_shape=cnn_output_shape,
            embed_dim=64,
            n_heads=8,
            n_layers=3,
            n_output_channels=c_trunk,
            smolgen_compress_dim=8,
            smolgen_shared_dim=32,
            feed_forward_multiplier=1.0
            )

        return ModelConfig(
            shape_info_dict=shape_info_dict,

            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),

            blocks=[
                ModuleSpec(type='ResBlock', args=['block1', c_trunk, c_mid]),
                ModuleSpec(type='TransformerBlock', args=[transformer_block_params]),
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
class Connect4Spec(GameSpec):
    name = 'c4'
    extra_runtime_deps = ['extra_deps/connect4/c4solver',
                          'extra_deps/connect4/7x6.book']
    model_configs = {
        'b7_c128': CNN_b7_c128,
        'transformer': Transformer,
        'b7_c128_beta0': CNN_b7_c128_beta0,
        'beta0': CNN_b7_c128_beta0,
        'default': CNN_b7_c128,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 21)
    ref_neighborhood_size = 21

    training_params = TrainingParams(
        target_sample_rate=8,
        minibatches_per_epoch=500,
        minibatch_size=100,
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
            first_run=500.0,
            benchmark=100.0,
            ),
        eval_error_threshold=50.0,
        n_games_per_self_evaluation=100,
        n_games_per_evaluation=1000,
    )


Connect4 = Connect4Spec()
