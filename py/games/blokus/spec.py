from games.game_spec import GameSpec
from shared.basic_types import ShapeInfoCollection
from shared.loss_term import BasicLossTerm, LossTerm
from shared.model_config import ModelConfig, ModelConfigGenerator, ModuleSpec
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams

from torch import optim

from dataclasses import dataclass
import math
from typing import List


class CNN_b20_c128(ModelConfigGenerator):
    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes
        target_shapes = head_shape_info_collection.target_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        action_value_shape = head_shapes['action_value'].shape
        ownership_shape = target_shapes['ownership'].shape
        score_shape = target_shapes['score'].shape
        unplayed_pieces_shape = target_shapes['unplayed_pieces'].shape

        c_trunk = 128
        c_mid = 128
        c_gpool = 32
        c_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256
        c_score_margin_hidden = 32
        n_score_margin_hidden = 32
        c_ownership_hidden = 64
        c_unplayed_pieces_hidden = 32
        n_unplayed_pieces_hidden = 32

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            trunk1=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=4,
                parents=['stem']
            ),
            trunk2=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[trunk_shape, c_gpool, res_mid_shape],
                parents=['trunk1']
            ),
            trunk3=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=4,
                parents=['trunk2']
            ),
            trunk4=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[trunk_shape, c_gpool, res_mid_shape],
                parents=['trunk3']
            ),
            trunk5=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=4,
                parents=['trunk4']
            ),
            trunk6=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[trunk_shape, c_gpool, res_mid_shape],
                parents=['trunk5']
            ),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=5,
                parents=['trunk6']
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
            score=ModuleSpec(
                type='ScoreHead',
                args=[trunk_shape, c_score_margin_hidden, n_score_margin_hidden, score_shape],
                parents=['trunk']
            ),
            ownership=ModuleSpec(
                type='OwnershipHead',
                args=[trunk_shape, c_ownership_hidden, ownership_shape],
                parents=['trunk']
            ),
            unplayed_pieces=ModuleSpec(
                # TODO: GeneralLogitHead doesn't exist anymore - fix this
                type='GeneralLogitHead',
                args=[trunk_shape, c_unplayed_pieces_hidden, n_unplayed_pieces_hidden,
                      unplayed_pieces_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 1.0),
            BasicLossTerm('score', 0.02),
            BasicLossTerm('ownership', 0.15),
            BasicLossTerm('unplayed_pieces', 0.3),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=6e-5, weight_decay=6e-5)


@dataclass
class BlokusSpec(GameSpec):
    name = 'blokus'
    num_players = 4
    model_configs = {
        'default': CNN_b20_c128,
        'b20_c128': CNN_b20_c128,
    }

    training_params = TrainingParams(
        target_sample_rate=8,
        minibatches_per_epoch=256,
        minibatch_size=64,
    )

    training_options = {
        '--mean-noisy-moves': 8,
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


Blokus = BlokusSpec()
