from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.basic_types import ShapeInfoCollection
from shared.loss_term import BasicLossTerm, LossTerm
from shared.model_config import ModelConfig, ModelConfigGenerator, ModuleSpec
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams

from torch import optim

from dataclasses import dataclass
import math
from typing import List


class CNN_b3_c32(ModelConfigGenerator):
    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        action_value_shape = head_shapes['action_value'].shape

        assert value_shape == (2,), value_shape

        c_trunk = 32
        c_mid = 32
        c_policy_hidden = 2
        c_action_value_hidden = 4
        c_value_hidden = 2
        n_value_hidden = 32

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=3,
                parents=['stem']
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            value=ModuleSpec(
                type='WinShareValueHead',
                args=[trunk_shape, c_value_hidden, n_value_hidden, value_shape],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[trunk_shape, c_action_value_hidden, action_value_shape],
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


@dataclass
class StochasticNimSpec(GameSpec):
    name = 'stochastic_nim'
    model_configs = {
        'default': CNN_b3_c32,
        'b3_c32': CNN_b3_c32,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 1)

    training_params = TrainingParams(
        target_sample_rate=16,
        minibatches_per_epoch=64,
    )

    rating_params = RatingParams(
        rating_player_options=RatingPlayerOptions(
            num_search_threads=4,
            num_iterations=25,
        ),
        default_target_elo_gap=DefaultTargetEloGap(
            first_run=25.0,
            benchmark=5.0,
        ),
        eval_error_threshold=5.0,
        n_games_per_self_evaluation=100,
        n_games_per_evaluation=1000,
    )


StochasticNim = StochasticNimSpec()
