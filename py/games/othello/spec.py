from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.basic_types import ShapeInfoCollection
from shared.loss_term import BasicLossTerm, LossTerm
from shared.model_config import ModelConfig, ModelConfigGenerator, ModuleSpec
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams
from shared.transformer_modules import TransformerBlockParams

from torch import optim

from dataclasses import dataclass
import math
from typing import List


class CNN_b9_c128(ModelConfigGenerator):
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
        score_margin_shape = target_shapes['score_margin'].shape
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

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),
            trunk1=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=4,
                parents=['stem']
            ),
            trunk2=ModuleSpec(
                type='ResBlockWithGlobalPooling',
                args=[c_trunk, c_mid, c_gpool],
                parents=['trunk1']
            ),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[c_trunk, c_mid],
                repeat=4,
                parents=['trunk2']
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
            score_margin=ModuleSpec(
                type='ScoreHead',
                args=[c_trunk, c_score_margin_hidden, n_score_margin_hidden, score_margin_shape],
                parents=['trunk']
            ),
            ownership=ModuleSpec(
                type='OwnershipHead',
                args=[c_trunk, c_ownership_hidden, ownership_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 2.0),
            BasicLossTerm('opp_policy', 0.03),
            BasicLossTerm('score_margin', 0.02),
            BasicLossTerm('ownership', 0.15),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=6e-5, weight_decay=6e-5)


class Transformer(ModelConfigGenerator):
    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes
        target_shapes = head_shape_info_collection.target_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        action_value_shape = head_shapes['action_value'].shape
        score_margin_shape = target_shapes['score_margin'].shape
        ownership_shape = target_shapes['ownership'].shape
        board_shape = input_shape[1:]
        board_size = math.prod(board_shape)

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        cnn_output_shape = (c_trunk, *board_shape)

        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256

        c_score_margin_hidden = 32
        n_score_margin_hidden = 32
        c_ownership_hidden = 64

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
            score_margin=ModuleSpec(
                type='ScoreHead',
                args=[c_trunk, c_score_margin_hidden, n_score_margin_hidden, score_margin_shape],
                parents=['trunk']
            ),
            ownership=ModuleSpec(
                type='OwnershipHead',
                args=[c_trunk, c_ownership_hidden, ownership_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 2.0),
            BasicLossTerm('opp_policy', 0.03),
            BasicLossTerm('score_margin', 0.02),
            BasicLossTerm('ownership', 0.15),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=5e-4, weight_decay=6e-5)


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
        'transformer': Transformer,
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
