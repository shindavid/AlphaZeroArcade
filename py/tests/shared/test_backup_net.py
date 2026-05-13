"""
Tests for BackupNet, the new backup-related heads (StaticLatentHead, ActionLatentHead,
ChildEmbeddingHead, AccumulatorHead), and the orphan-`nnue/*`-initializer export pipeline
in Model.save_model.
"""
import os
import tempfile
import unittest

import numpy as np
import onnx
import torch

from shared.backup_net import BackupNet
from shared.basic_types import ShapeInfo, ShapeInfoCollection
from shared.model import Model
from shared.model_config import ModelConfig, ModuleSpec
from shared.net_modules import (
    AccumulatorHead, ActionLatentHead, ChildEmbeddingHead, StaticLatentHead,
)


class TestBackupNet(unittest.TestCase):

    def test_forward_shape(self):
        net = BackupNet(value_dim=3, static_latent_dim=8, embed_dim=16, layer1_dim=12,
                        layer2_dim=6)
        B = 4
        accumulator = torch.randn(B, 16)
        z_s = torch.randn(B, 8)
        S_baseline = torch.softmax(torch.randn(B, 3), dim=-1)
        Ws_baseline = torch.rand(B)
        out = net(accumulator, z_s, S_baseline, Ws_baseline)
        self.assertEqual(tuple(out.shape), (B, 4))

    def test_collect_graph_initializers_keys(self):
        net = BackupNet(value_dim=3, static_latent_dim=8, embed_dim=16, layer1_dim=12,
                        layer2_dim=6)
        out = {}
        net.collect_graph_initializers(out)
        expected = {
            'layer1.weight', 'layer1.bias',
            'layer2.weight', 'layer2.bias',
            'out.weight', 'out.bias',
        }
        self.assertEqual(set(out.keys()), expected)
        # layer1 input = embed (16) + d_s (8) + value_dim (3) + 1
        self.assertEqual(out['layer1.weight'].shape, (12, 16 + 8 + 3 + 1))
        self.assertEqual(out['layer1.bias'].shape, (12,))
        self.assertEqual(out['layer2.weight'].shape, (6, 12))
        self.assertEqual(out['layer2.bias'].shape, (6,))
        self.assertEqual(out['out.weight'].shape, (4, 6))
        self.assertEqual(out['out.bias'].shape, (4,))
        for arr in out.values():
            self.assertEqual(arr.dtype, np.float32)


class TestStaticLatentHead(unittest.TestCase):

    def test_forward_shape(self):
        head = StaticLatentHead(trunk_shape=(8, 6, 7), c_hidden=4, latent_dim=12)
        x = torch.randn(3, 8, 6, 7)
        out = head(x)
        self.assertEqual(tuple(out.shape), (3, 12))

    def test_default_loss_function_is_none(self):
        head = StaticLatentHead(trunk_shape=(8, 6, 7), c_hidden=4, latent_dim=12)
        self.assertIsNone(head.default_loss_function())


class TestActionLatentHead(unittest.TestCase):

    def test_forward_shape(self):
        head = ActionLatentHead(trunk_shape=(8, 6, 7), c_hidden=4, output_shape=(7, 5))
        x = torch.randn(3, 8, 6, 7)
        out = head(x)
        self.assertEqual(tuple(out.shape), (3, 7, 5))

    def test_default_loss_function_is_none(self):
        head = ActionLatentHead(trunk_shape=(8, 6, 7), c_hidden=4, output_shape=(7, 5))
        self.assertIsNone(head.default_loss_function())


class TestChildEmbeddingHead(unittest.TestCase):

    def _make(self, A=7, za=5, embed=11):
        return ChildEmbeddingHead(
            child_stats_shape=(A, 6),
            action_latent_shape=(A, za),
            embed_dim=embed,
        ), A, za, embed

    def _random_inputs(self, B, A, za, p_zero_count=0):
        # child_stats: (B, A, 6) = [Qs, Ws, N, P, AVs, AUs]
        Qs = torch.randn(B, A)
        Ws = torch.rand(B, A)
        N = torch.randint(0, 5, (B, A)).float()
        P = torch.full((B, A), 0.1)
        if p_zero_count > 0:
            P[:, A - p_zero_count:] = 0.0
        AVs = torch.randn(B, A)
        AUs = torch.rand(B, A)
        child_stats = torch.stack([Qs, Ws, N, P, AVs, AUs], dim=-1)
        action_latent = torch.randn(B, A, za)
        return child_stats, action_latent

    def test_forward_shape(self):
        head, A, za, embed = self._make()
        B = 3
        child_stats, action_latent = self._random_inputs(B, A, za)
        out = head(child_stats, action_latent)
        self.assertEqual(tuple(out.shape), (B, A, embed))

    def test_zero_policy_actions_are_masked_out(self):
        head, A, za, embed = self._make()
        B = 2
        child_stats, action_latent = self._random_inputs(B, A, za, p_zero_count=2)
        out = head(child_stats, action_latent)
        self.assertTrue(torch.all(out[:, A - 2:, :] == 0))
        self.assertFalse(torch.all(out[:, :A - 2, :] == 0))

    def test_collect_graph_initializers(self):
        head, A, za, embed = self._make()
        out = {}
        head.collect_graph_initializers(out)
        self.assertEqual(set(out.keys()), {'child_embed.weight', 'child_embed.bias'})
        per_child_in = 6 + za
        self.assertEqual(out['child_embed.weight'].shape, (embed, per_child_in))
        self.assertEqual(out['child_embed.bias'].shape, (embed,))


class TestAccumulatorHead(unittest.TestCase):

    def test_forward_shape_and_no_params(self):
        head = AccumulatorHead()
        e = torch.randn(3, 7, 11)
        out = head(e)
        self.assertEqual(tuple(out.shape), (3, 11))
        self.assertEqual(list(head.parameters()), [])

    def test_sum_pool(self):
        head = AccumulatorHead()
        e = torch.ones(2, 4, 3)
        out = head(e)
        self.assertTrue(torch.allclose(out, torch.full((2, 3), 4.0)))


def _build_model_with_backup():
    """
    Build a minimal Model wired up with the full backup-net stack for save_model testing.
    """
    input_shape = (2, 6, 7)
    trunk_shape = (4, 6, 7)
    A = 7
    av_dim = 2
    au_dim = 1
    za_dim = 3
    embed_dim = 8
    static_latent_dim = 5

    config = ModelConfig.create(
        external_inputs=['value_baseline', 'value_uncertainty_baseline', 'child_stats'],
        stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
        policy=ModuleSpec(type='PolicyHead',
                          args=[trunk_shape, 2, (A,)], parents=['stem']),
        value=ModuleSpec(type='WinLossDrawValueHead',
                         args=[trunk_shape, 1, 16], parents=['stem']),
        action_value=ModuleSpec(type='WinShareActionValueHead',
                                args=[trunk_shape, 2, (A, av_dim)], parents=['stem']),
        action_value_uncertainty=ModuleSpec(
            type='ActionValueUncertaintyHead',
            args=[trunk_shape, 2, (A, au_dim)], parents=['stem']),
        static_latent=ModuleSpec(type='StaticLatentHead',
                                 args=[trunk_shape, 2, static_latent_dim], parents=['stem']),
        action_latent=ModuleSpec(type='ActionLatentHead',
                                 args=[trunk_shape, 2, (A, za_dim)], parents=['stem']),
        child_embedding=ModuleSpec(
            type='ChildEmbeddingHead',
            args=[(A, 6), (A, za_dim), embed_dim],
            parents=['child_stats', 'action_latent']),
        accumulator=ModuleSpec(type='AccumulatorHead', parents=['child_embedding']),
        backup_net=ModuleSpec(
            type='BackupNet',
            kwargs={
                'value_dim': 3,
                'static_latent_dim': static_latent_dim,
                'embed_dim': embed_dim,
                'layer1_dim': 6,
                'layer2_dim': 4,
            },
            parents=['accumulator', 'static_latent', 'value_baseline', 'value_uncertainty_baseline']),
    )
    model = Model(config)
    shape_info = ShapeInfoCollection(
        input_shapes={
            'input': ShapeInfo('input', 0, input_shape),
            'child_stats': ShapeInfo('child_stats', 1, (A, 6)),
            'value_baseline': ShapeInfo('value_baseline', 2, (3,)),
            'value_uncertainty_baseline': ShapeInfo('value_uncertainty_baseline', 3, (1,)),
        },
        target_shapes={},
        head_shapes={
            'policy': ShapeInfo('policy', 0, (A,)),
            'value': ShapeInfo('value', 1, (3,)),
            'action_value': ShapeInfo('action_value', 2, (A, av_dim)),
            'action_value_uncertainty': ShapeInfo(
                'action_value_uncertainty', 3, (A, au_dim)),
            'static_latent': ShapeInfo('static_latent', 4, (static_latent_dim,)),
            'action_latent': ShapeInfo('action_latent', 5, (A, za_dim)),
            'child_embedding': ShapeInfo('child_embedding', 6, (A, embed_dim)),
            'accumulator': ShapeInfo('accumulator', 7, (embed_dim,)),
        },
    )
    return model, shape_info, embed_dim


class TestModelNNUEExport(unittest.TestCase):

    def test_collect_nnue_initializers_via_model(self):
        model, _, _ = _build_model_with_backup()
        nnue = model._collect_nnue_initializers()
        # ChildEmbeddingHead contributes child_embed.*
        self.assertIn('child_embed.weight', nnue)
        self.assertIn('child_embed.bias', nnue)
        # BackupNet contributes layer1/2/out
        for k in ('layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias',
                  'out.weight', 'out.bias'):
            self.assertIn(k, nnue)

    def test_save_model_embeds_nnue_initializers(self):
        model, shape_info, _ = _build_model_with_backup()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'model.onnx')
            model.save_model(onnx_path, shape_info)
            saved = onnx.load(onnx_path)

        init_names = {init.name for init in saved.graph.initializer}
        nnue_names = {n for n in init_names if n.startswith('nnue/')}

        expected = {
            'nnue/child_embed.weight', 'nnue/child_embed.bias',
            'nnue/layer1.weight', 'nnue/layer1.bias',
            'nnue/layer2.weight', 'nnue/layer2.bias',
            'nnue/out.weight', 'nnue/out.bias',
        }
        self.assertEqual(nnue_names, expected)

        # Confirm the embedded values match the live module weights.
        live = model._collect_nnue_initializers()
        by_name = {init.name: init for init in saved.graph.initializer}
        for k, arr in live.items():
            embedded = onnx.numpy_helper.to_array(by_name[f'nnue/{k}'])
            np.testing.assert_array_equal(embedded, arr)

    def test_model_without_backup_collects_nothing(self):
        input_shape = (2, 6, 7)
        trunk_shape = (4, 6, 7)
        config = ModelConfig.create(
            stem=ModuleSpec(type='ConvBlock', args=[input_shape, trunk_shape]),
            policy=ModuleSpec(type='PolicyHead',
                              args=[trunk_shape, 2, (7,)], parents=['stem']),
            value=ModuleSpec(type='WinLossDrawValueHead',
                             args=[trunk_shape, 1, 32], parents=['stem']),
            action_value=ModuleSpec(type='WinShareActionValueHead',
                                    args=[trunk_shape, 2, (7, 2)], parents=['stem']),
        )
        model = Model(config)
        self.assertEqual(model._collect_nnue_initializers(), {})


if __name__ == '__main__':
    unittest.main()
