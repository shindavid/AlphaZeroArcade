from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.basic_types import ShapeInfoDict
from shared.loss_term import BasicLossTerm, LossTerm
from shared.model_config import ModelConfig, ModelConfigGenerator, ModuleSpec
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams
from shared.transformer_modules import TransformerBlockParams

from torch import optim

from dataclasses import dataclass
import math
from typing import List


class CNN_b3_c32(ModelConfigGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
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

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=3,
                parents=['stem']
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            value=ModuleSpec(
                type='WinLossDrawValueHead',
                args=[board_size, c_trunk, c_value_hidden, n_value_hidden],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[board_size, c_trunk, c_action_value_hidden, action_value_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 1.0),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=6e-5, weight_decay=6e-5)


class Mini(ModelConfigGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
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
        c_policy_hidden = 1
        c_action_value_hidden = 1
        c_value_hidden = 1
        n_value_hidden = 1

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_policy_hidden, policy_shape],
                parents=['stem']
            ),
            value=ModuleSpec(
                type='WinLossDrawValueHead',
                args=[board_size, c_trunk, c_value_hidden, n_value_hidden],
                parents=['stem']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[board_size, c_trunk, c_action_value_hidden, action_value_shape],
                parents=['stem']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 1.0),
        ]

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

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),
            pre_trunk=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                parents=['stem']
            ),
            trunk=ModuleSpec(
                type='TransformerBlock',
                args=[transformer_block_params],
                parents=['pre_trunk']
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            value=ModuleSpec(
                type='WinLossDrawValueHead',
                args=[board_size, c_trunk, c_value_hidden, n_value_hidden],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[board_size, c_trunk, c_action_value_hidden, action_value_shape],
                parents=['trunk']
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[board_size, c_trunk, c_opp_policy_hidden, policy_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 1.0),
            BasicLossTerm('opp_policy', 0.15),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=1e-3, weight_decay=6e-5)


@dataclass
class TicTacToeSpec(GameSpec):
    name = 'tictactoe'
    model_configs = {
        'default': CNN_b3_c32,
        'b3_c32': CNN_b3_c32,
        'mini': Mini,
        'transformer': Transformer,
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
        n_games_per_self_evaluation=100,
        n_games_per_evaluation=1000,
    )



TicTacToe = TicTacToeSpec()
