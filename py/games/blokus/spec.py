from games.game_spec import GameSpec
from shared.net_modules import ModelConfig, ModelConfigGenerator, ModuleSpec, ShapeInfoDict
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams

from dataclasses import dataclass
import math
from torch import optim


class CNN_b20_c128(ModelConfigGenerator):
    @staticmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
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
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256
        c_score_margin_hidden = 32
        n_score_margin_hidden = 32
        c_ownership_hidden = 64
        c_unplayed_pieces_hidden = 32
        n_unplayed_pieces_hidden = 32

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),
            trunk1=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=4,
                parent='stem'
            ),
            trunk2=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[c_trunk, c_mid, c_gpool],
                parent='trunk1'
            ),
            trunk3=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=4,
                parent='trunk2'
            ),
            trunk4=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[c_trunk, c_mid, c_gpool],
                parent='trunk3'
            ),
            trunk5=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=4,
                parent='trunk4'
            ),
            trunk6=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[c_trunk, c_mid, c_gpool],
                parent='trunk5'
            ),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=5,
                parent='trunk6'
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape],
                head=True,
                parent='trunk'
            ),
            value=ModuleSpec(
                type='WinShareValueHead',
                args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                      value_shape],
                head=True,
                parent='trunk'
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=['action_value', board_size, c_trunk, c_action_value_hidden,
                      action_value_shape],
                head=True,
                parent='trunk'
            ),
            score=ModuleSpec(
                type='ScoreHead',
                args=['score', c_trunk, c_score_margin_hidden, n_score_margin_hidden, score_shape],
                head=True,
                parent='trunk'
            ),
            ownership=ModuleSpec(
                type='OwnershipHead',
                args=['ownership', c_trunk, c_ownership_hidden, ownership_shape],
                head=True,
                parent='trunk'
            ),
            unplayed_pieces=ModuleSpec(
                type='GeneralLogitHead',
                args=['unplayed_pieces', c_trunk, c_unplayed_pieces_hidden,
                      n_unplayed_pieces_hidden, unplayed_pieces_shape],
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
            'score': 0.02,
            'ownership': 0.15,
            'unplayed_pieces': 0.3,
        }

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
