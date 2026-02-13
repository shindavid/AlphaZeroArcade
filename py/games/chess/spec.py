from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.basic_types import SearchParadigm, ShapeInfoCollection
from shared.loss_term import BasicLossTerm, LossTerm, ValueUncertaintyLossTerm
from shared.model_config import ModelConfig, ModelConfigGenerator, ModuleSequenceSpec, ModuleSpec
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams
from shared.transformer_modules import TransformerBlockParams

from torch import optim

from dataclasses import dataclass
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

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        c_gpool = 32
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256
        c_score_hidden = 32
        n_score_hidden = 32
        c_ownership_hidden = 64

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            trunk=ModuleSequenceSpec(
                ModuleSpec(
                    type='ResBlock',
                    args=[trunk_shape, res_mid_shape],
                    repeat=4,
                ),
                ModuleSpec(
                    type='ResBlockWithGlobalPooling',
                    args=[trunk_shape, c_gpool, res_mid_shape],
                ),
                ModuleSpec(
                    type='ResBlock',
                    args=[trunk_shape, res_mid_shape],
                    repeat=4,
                ),
                parents=['stem']
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            value=ModuleSpec(
                type='WinLossDrawValueHead',
                args=[trunk_shape, c_value_hidden, n_value_hidden],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[trunk_shape, c_action_value_hidden, action_value_shape],
                parents=['trunk']
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_opp_policy_hidden, policy_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 5.0),
            BasicLossTerm('opp_policy', 0.03),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=6e-5, weight_decay=6e-5)


class CNN_b9_c128_beta0(ModelConfigGenerator):
    search_paradigm: SearchParadigm = SearchParadigm.BetaZero

    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes
        target_shapes = head_shape_info_collection.target_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        action_value_shape = head_shapes['action_value'].shape
        value_uncertainty_shape = head_shapes['value_uncertainty'].shape
        action_value_uncertainty_shape = head_shapes['action_value_uncertainty'].shape
        ownership_shape = target_shapes['ownership'].shape
        score_margin_shape = target_shapes['score_margin'].shape

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        c_gpool = 32
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256
        c_score_hidden = 32
        n_score_hidden = 32
        c_ownership_hidden = 64
        c_value_uncertainty_hidden = 1
        c_action_value_uncertainty_hidden = 2
        n_value_uncertainty_hidden = 256

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            trunk=ModuleSequenceSpec(
                ModuleSpec(
                    type='ResBlock',
                    args=[trunk_shape, res_mid_shape],
                    repeat=4,
                ),
                ModuleSpec(
                    type='ResBlockWithGlobalPooling',
                    args=[trunk_shape, c_gpool, res_mid_shape],
                ),
                ModuleSpec(
                    type='ResBlock',
                    args=[trunk_shape, res_mid_shape],
                    repeat=4,
                ),
                parents=['stem']
            ),

            policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            value=ModuleSpec(
                type='WinLossDrawValueHead',
                args=[trunk_shape, c_value_hidden, n_value_hidden],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[trunk_shape, c_action_value_hidden, action_value_shape],
                parents=['trunk']
            ),
            value_uncertainty=ModuleSpec(
                type='ValueUncertaintyHead',
                args=[trunk_shape, value_shape, c_value_uncertainty_hidden,
                      n_value_uncertainty_hidden, value_uncertainty_shape],
                parents=['trunk', 'value']
            ),
            action_value_uncertainty=ModuleSpec(
                type='ActionValueUncertaintyHead',
                args=[trunk_shape, c_action_value_uncertainty_hidden,
                      action_value_uncertainty_shape],
                parents=['trunk']
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_opp_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            score_margin=ModuleSpec(
                type='ScoreHead',
                args=[trunk_shape, c_score_hidden, n_score_hidden,
                      score_margin_shape],
                parents=['trunk']
            ),
            ownership=ModuleSpec(
                type='OwnershipHead',
                args=[trunk_shape, c_ownership_hidden, ownership_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 5.0),
            ValueUncertaintyLossTerm('value_uncertainty', 10.0),  # currently not used in c++
            BasicLossTerm('action_value_uncertainty', 150.0),
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

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256
        c_score_hidden = 32
        n_score_hidden = 32
        c_ownership_hidden = 64

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        transformer_block_params = TransformerBlockParams(
            input_shape=trunk_shape,
            embed_dim=64,
            n_heads=8,
            n_layers=3,
            n_output_channels=c_trunk,
            smolgen_compress_dim=8,
            smolgen_shared_dim=32,
            feed_forward_multiplier=1.0
            )

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            pre_trunk=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
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
                args=[trunk_shape, c_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            value=ModuleSpec(
                type='WinLossDrawValueHead',
                args=[trunk_shape, c_value_hidden, n_value_hidden],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[trunk_shape, c_action_value_hidden, action_value_shape],
                parents=['trunk']
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_opp_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            score_margin=ModuleSpec(
                type='ScoreHead',
                args=[trunk_shape, c_score_hidden, n_score_hidden, score_margin_shape],
                parents=['trunk']
            ),
            ownership=ModuleSpec(
                type='OwnershipHead',
                args=[trunk_shape, c_ownership_hidden, ownership_shape],
                parents=['trunk']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 5.0),
            BasicLossTerm('opp_policy', 0.03),
            BasicLossTerm('score_margin', 0.02),
            BasicLossTerm('ownership', 0.15),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=5e-4, weight_decay=6e-5)


@dataclass
class ChessSpec(GameSpec):
    name = 'chess'
    extra_runtime_deps = []
    model_configs = {
        'default': CNN_b9_c128,
        'b9_c128': CNN_b9_c128,
        'transformer': Transformer,
        'b9_c128_beta0': CNN_b9_c128_beta0,
        'beta0': CNN_b9_c128_beta0,
    }
    reference_player_family = None
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


Chess = ChessSpec()
