from games.game_spec import GameSpec
from shared.net_modules import ModelConfig, ModelConfigGenerator, ModuleSpec, ShapeInfoDict
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams
from shared.transformer_modules import TransformerBlockParams

from dataclasses import dataclass
import math
from torch import optim


class CNN_b11_c32(ModelConfigGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
        input_shape = shape_info_dict['input'].shape
        policy_shape = shape_info_dict['policy'].shape
        value_shape = shape_info_dict['value'].shape
        action_value_shape = shape_info_dict['action_value'].shape
        board_shape = input_shape[1:]
        board_size = math.prod(board_shape)

        assert value_shape == (2,), value_shape

        c_trunk = 32
        c_mid = 32
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=11,
                parent='stem'
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_policy_hidden, policy_shape],
                head=True,
                parent='trunk'
            ),
            value=ModuleSpec(
                type='WinLossValueHead',
                args=[board_size, c_trunk, c_value_hidden, n_value_hidden],
                head=True,
                parent='trunk'
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[board_size, c_trunk, c_action_value_hidden, action_value_shape],
                head=True,
                parent='trunk'
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_opp_policy_hidden, policy_shape],
                head=True,
                parent='trunk'
            ),
        )

    @staticmethod
    def loss_weights():
        return {
            'policy': 1.0,
            'value': 1.5,
            'action_value': 1.0,
            'opp_policy': 0.15,
        }

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=6e-5, weight_decay=6e-5)


class Transformer(ModelConfigGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
        input_shape = shape_info_dict['input'].shape
        policy_shape = shape_info_dict['policy'].shape
        value_shape = shape_info_dict['value'].shape
        action_value_shape = shape_info_dict['action_value'].shape
        board_shape = input_shape[1:]
        board_size = math.prod(board_shape)

        assert value_shape == (2,), value_shape

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

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),
            pre_trunk=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=2,
                parent='stem'
            ),
            trunk=ModuleSpec(
                type='TransformerBlock',
                args=[transformer_block_params],
                parent='pre_trunk'
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_policy_hidden, policy_shape],
                head=True,
                parent='trunk'
            ),
            value=ModuleSpec(
                type='WinLossValueHead',
                args=[board_size, c_trunk, c_value_hidden, n_value_hidden],
                head=True,
                parent='trunk'
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[board_size, c_trunk, c_action_value_hidden, action_value_shape],
                head=True,
                parent='trunk'
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_opp_policy_hidden, policy_shape],
                head=True,
                parent='trunk'
            ),
        )

    @staticmethod
    def loss_weights():
        return {
            'policy': 1.0,
            'value': 1.5,
            'action_value': 1.0,
            'opp_policy': 0.15,
        }

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=5e-4, weight_decay=6e-5)


@dataclass
class HexSpec(GameSpec):
    name = 'hex'
    model_configs = {
        'b11_c32': CNN_b11_c32,
        'transformer': Transformer,
        'default': CNN_b11_c32,
    }

    training_params = TrainingParams(
        target_sample_rate=8,
        minibatches_per_epoch=500,
        minibatch_size=100,
    )

    training_options = {
        '--mean-noisy-moves': 6,
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


Hex = HexSpec()
