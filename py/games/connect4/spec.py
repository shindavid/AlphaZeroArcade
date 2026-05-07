from games.game_spec import GameSpec, ReferencePlayerFamily
from shared.basic_types import SearchParadigm, ShapeInfoCollection
from shared.loss_term import BasicLossTerm, LossTerm, ValueUncertaintyLossTerm
from shared.model_config import ModelConfig, ModelConfigGenerator, ModuleSpec
from shared.rating_params import DefaultTargetEloGap, RatingParams, RatingPlayerOptions
from shared.training_params import TrainingParams
from shared.transformer_modules import TransformerBlockParams

from torch import optim

from dataclasses import dataclass
from typing import List


class CNN_b7_c128(ModelConfigGenerator):
    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        action_value_shape = head_shapes['action_value'].shape

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=7,
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


class CNN_b7_c128_beta0(ModelConfigGenerator):
    spec_name: str = 'beta0'
    paradigm: SearchParadigm = SearchParadigm.BetaZero

    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        uncertainty_shape = head_shapes['uncertainty'].shape
        action_value_shape = head_shapes['action_value'].shape
        action_uncertainty_shape = head_shapes['action_uncertainty'].shape

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256
        c_uncertainty_hidden = 1
        n_uncertainty_hidden = 256
        c_action_uncertainty_hidden = 2

        # BackupNet hyperparameters (BetaZero CPU-side NNUE).
        c_static_latent_hidden = 16
        static_latent_dim = 32
        c_action_latent_hidden = 4
        action_latent_dim = 8
        backup_embed_dim = 64
        backup_layer1_dim = 32
        backup_layer2_dim = 16

        action_latent_shape = (policy_shape[0], action_latent_dim)

        board_shape = input_shape[1:]
        trunk_shape = (c_trunk, *board_shape)
        res_mid_shape = (c_mid, *board_shape)

        # Heads supporting the BetaZero CPU-side NNUE backup:
        #
        #   * static_latent (z_s)    - per-node global latent
        #   * action_latent (z_a)    - per-action latent
        #   * child_embedding (e_i)  - per-child embeddings, NNUE-style subtract-add target
        #   * accumulator            - sum-pool of child_embedding over actions
        #   * backup_net             - dense layers (accumulator, z_s, Qs*, Ws*) -> (Q, W)
        #
        # backup_net is declared like any other DAG node; its parents include the external
        # inputs `input_Qs_star` and `input_Ws_star`. It is not part of the inference graph (no
        # inference target depends on it), but its weights are exported as orphan `nnue/*`
        # initializers via Model.save_model's walk over the un-trimmed model. See
        # docs/BetaZero.pdf, Sections 4.2, 4.3, and 7.1.
        #
        # TODO: Wire the BackupNet into training (BackupLossTerm). Currently the backup-related
        # heads and the BackupNet weights produce no loss signal, so they remain at their PyTorch
        # defaults.
        # TODO: Decide how `input_Qs_star`, `input_Ws_star`, and `input_child_stats` are populated
        # TODO: Decide how `input_Qs_star`, `input_Ws_star`, and `input_child_stats` are populated
        # (game-log columns vs. on-the-fly computation with stop-gradient à la
        # ValueUncertaintyLossTerm).
        # TODO: Remove the hardcoded latent / embed dims here once the C++ side exposes them via
        # head_shapes / FFI; spec.py should derive them from head_shape_info_collection instead.

        return ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            trunk=ModuleSpec(
                type='ResBlock',
                args=[trunk_shape, res_mid_shape],
                repeat=7,
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
            value_uncertainty=ModuleSpec(
                type='ValueUncertaintyHead',
                args=[trunk_shape, c_uncertainty_hidden, n_uncertainty_hidden, uncertainty_shape],
                parents=['trunk']
            ),
            action_value=ModuleSpec(
                type='WinShareActionValueHead',
                args=[trunk_shape, c_action_value_hidden, action_value_shape],
                parents=['trunk']
            ),
            action_value_uncertainty=ModuleSpec(
                type='ActionValueUncertaintyHead',
                args=[trunk_shape, c_action_uncertainty_hidden, action_uncertainty_shape],
                parents=['trunk']
            ),
            opp_policy=ModuleSpec(
                type='PolicyHead',
                args=[trunk_shape, c_opp_policy_hidden, policy_shape],
                parents=['trunk']
            ),
            static_latent=ModuleSpec(
                type='StaticLatentHead',
                args=[trunk_shape, c_static_latent_hidden, static_latent_dim],
                parents=['trunk']
            ),
            action_latent=ModuleSpec(
                type='ActionLatentHead',
                args=[trunk_shape, c_action_latent_hidden, action_latent_shape],
                parents=['trunk']
            ),
            child_embedding=ModuleSpec(
                type='ChildEmbeddingHead',
                args=[(policy_shape[0], 6), action_latent_shape, backup_embed_dim],
                parents=['input_child_stats', 'action_latent']
            ),
            accumulator=ModuleSpec(
                type='AccumulatorHead',
                parents=['child_embedding']
            ),
            backup_net=ModuleSpec(
                type='BackupNet',
                kwargs={
                    'static_latent_dim': static_latent_dim,
                    'embed_dim': backup_embed_dim,
                    'layer1_dim': backup_layer1_dim,
                    'layer2_dim': backup_layer2_dim,
                },
                parents=['accumulator', 'static_latent', 'input_Qs_star', 'input_Ws_star']
            ),
        )

    @staticmethod
    def loss_terms() -> List[LossTerm]:
        # TODO: add a BackupLossTerm for the BackupNet outputs (Q, W). The static_latent head has
        # no direct loss; it is trained via gradient flowing back through the BackupNet.
        return [
            BasicLossTerm('policy', 1.0),
            BasicLossTerm('value', 1.5),
            BasicLossTerm('action_value', 5.0),
            BasicLossTerm('opp_policy', 0.03),
            ValueUncertaintyLossTerm('value_uncertainty', 32.0),
            BasicLossTerm('action_value_uncertainty', 32.0),
        ]

    @staticmethod
    def optimizer(params) -> optim.Optimizer:
        return optim.RAdam(params, lr=6e-5, weight_decay=6e-5)


class Transformer(ModelConfigGenerator):
    @staticmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        input_shapes = head_shape_info_collection.input_shapes
        head_shapes = head_shape_info_collection.head_shapes

        input_shape = input_shapes['input'].shape
        policy_shape = head_shapes['policy'].shape
        value_shape = head_shapes['value'].shape
        action_value_shape = head_shapes['action_value'].shape

        assert value_shape == (3,), value_shape

        c_trunk = 128
        c_mid = 128
        c_policy_hidden = 2
        c_opp_policy_hidden = 2
        c_action_value_hidden = 2
        c_value_hidden = 1
        n_value_hidden = 256

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
        return optim.RAdam(params, lr=5e-4, weight_decay=6e-5)


@dataclass
class Connect4Spec(GameSpec):
    name = 'c4'
    extra_runtime_deps = ['extra_deps/connect4/c4solver',
                          'extra_deps/connect4/7x6.book']
    model_configs = {
        'b7_c128': CNN_b7_c128,
        'transformer': Transformer,
        'default': CNN_b7_c128,
        'beta0': CNN_b7_c128_beta0,
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
