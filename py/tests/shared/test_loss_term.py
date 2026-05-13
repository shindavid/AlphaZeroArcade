from shared.loss_term import (
    BackupLossTerm,
    Masker,
    BasicLossTerm,
    ValueUncertaintyLossTerm,
)

import torch
from torch import nn
from unittest.mock import MagicMock
import unittest


class TestMasker(unittest.TestCase):

    def test_no_y_names_no_masking(self):
        y_hat_dict = {'pred': torch.tensor([1.0, 2.0, 3.0])}
        masker = Masker(mask_dict={}, y_hat_dict=y_hat_dict, y_dict={})

        y_hats, ys = masker.get_y_hat_and_y(['pred'], [])
        self.assertEqual(len(y_hats), 1)
        self.assertEqual(len(ys), 0)
        torch.testing.assert_close(y_hats[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_single_mask_filters(self):
        t = torch.tensor([10.0, 20.0, 30.0, 40.0])
        mask = torch.tensor([True, False, True, False])
        masker = Masker(
            mask_dict={'target': mask},
            y_hat_dict={'pred': t.clone()},
            y_dict={'target': t.clone()},
        )

        y_hats, ys = masker.get_y_hat_and_y(['pred'], ['target'])
        self.assertEqual(y_hats[0].shape, (2,))
        self.assertEqual(ys[0].shape, (2,))
        torch.testing.assert_close(y_hats[0], torch.tensor([10.0, 30.0]))
        torch.testing.assert_close(ys[0], torch.tensor([10.0, 30.0]))

    def test_multiple_masks_anded(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask_a = torch.tensor([True, True, False, True])
        mask_b = torch.tensor([True, False, False, True])
        masker = Masker(
            mask_dict={'a': mask_a, 'b': mask_b},
            y_hat_dict={'pred': t.clone()},
            y_dict={'a': t.clone(), 'b': t.clone()},
        )

        y_hats, ys = masker.get_y_hat_and_y(['pred'], ['a', 'b'])
        # AND of masks: [True, False, False, True] → indices 0 and 3
        self.assertEqual(y_hats[0].shape, (2,))
        torch.testing.assert_close(y_hats[0], torch.tensor([1.0, 4.0]))

    def test_all_true_mask(self):
        t = torch.tensor([5.0, 6.0, 7.0])
        mask = torch.tensor([True, True, True])
        masker = Masker(
            mask_dict={'target': mask},
            y_hat_dict={'pred': t.clone()},
            y_dict={'target': t.clone()},
        )

        y_hats, ys = masker.get_y_hat_and_y(['pred'], ['target'])
        self.assertEqual(y_hats[0].shape, (3,))
        torch.testing.assert_close(y_hats[0], t)

    def test_all_false_mask(self):
        t = torch.tensor([5.0, 6.0, 7.0])
        mask = torch.tensor([False, False, False])
        masker = Masker(
            mask_dict={'target': mask},
            y_hat_dict={'pred': t.clone()},
            y_dict={'target': t.clone()},
        )

        y_hats, ys = masker.get_y_hat_and_y(['pred'], ['target'])
        self.assertEqual(y_hats[0].shape, (0,))
        self.assertEqual(ys[0].shape, (0,))

    def test_2d_tensors(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = torch.tensor([True, False, True])
        masker = Masker(
            mask_dict={'target': mask},
            y_hat_dict={'pred': t.clone()},
            y_dict={'target': t.clone()},
        )

        y_hats, ys = masker.get_y_hat_and_y(['pred'], ['target'])
        self.assertEqual(y_hats[0].shape, (2, 2))
        torch.testing.assert_close(y_hats[0], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    def test_multiple_y_hats(self):
        pred1 = torch.tensor([1.0, 2.0, 3.0])
        pred2 = torch.tensor([10.0, 20.0, 30.0])
        mask = torch.tensor([False, True, True])
        masker = Masker(
            mask_dict={'target': mask},
            y_hat_dict={'p1': pred1, 'p2': pred2},
            y_dict={'target': torch.zeros(3)},
        )

        y_hats, ys = masker.get_y_hat_and_y(['p1', 'p2'], ['target'])
        self.assertEqual(len(y_hats), 2)
        torch.testing.assert_close(y_hats[0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(y_hats[1], torch.tensor([20.0, 30.0]))

    def test_mask_not_mutated(self):
        mask = torch.tensor([True, False, True])
        original_mask = mask.clone()
        masker = Masker(
            mask_dict={'target': mask},
            y_hat_dict={'pred': torch.tensor([1.0, 2.0, 3.0])},
            y_dict={'target': torch.tensor([4.0, 5.0, 6.0])},
        )
        masker.get_y_hat_and_y(['pred'], ['target'])
        torch.testing.assert_close(mask, original_mask)


class TestLossTermInit(unittest.TestCase):

    def test_name_and_weight(self):
        lt = BasicLossTerm(head='value', weight=1.5)
        self.assertEqual(lt.name, 'value')
        self.assertEqual(lt.weight, 1.5)


class TestBasicLossTerm(unittest.TestCase):

    def test_init_lazy_state(self):
        lt = BasicLossTerm(head='value', weight=1.0)
        self.assertIsNone(lt._head)
        self.assertIsNone(lt._loss_fn)
        self.assertFalse(lt._use_policy_scaling)

    def _make_mock_model(self, loss_fn_type, policy_scaling=False):
        head = MagicMock()
        head.default_loss_function.return_value = loss_fn_type
        head.requires_policy_scaling.return_value = policy_scaling
        model = MagicMock()
        model.get_head.return_value = head
        return model

    def test_post_init_sets_loss_fn(self):
        model = self._make_mock_model(nn.MSELoss, policy_scaling=False)
        lt = BasicLossTerm(head='value', weight=1.0)
        lt.post_init(model)
        self.assertIsNotNone(lt._loss_fn)
        self.assertIsInstance(lt._loss_fn, nn.MSELoss)
        self.assertFalse(lt._use_policy_scaling)

    def test_compute_loss_no_scaling(self):
        model = self._make_mock_model(nn.MSELoss, policy_scaling=False)
        lt = BasicLossTerm(head='value', weight=1.0)
        lt.post_init(model)

        B = 4
        y_hat = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(B, dtype=torch.bool)

        masker = Masker(
            mask_dict={'value': mask},
            y_hat_dict={'value': y_hat},
            y_dict={'value': y},
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, 4)
        # MSE of identical tensors should be 0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_compute_loss_nonzero(self):
        model = self._make_mock_model(nn.MSELoss, policy_scaling=False)
        lt = BasicLossTerm(head='value', weight=1.0)
        lt.post_init(model)

        y_hat = torch.tensor([1.0, 2.0])
        y = torch.tensor([2.0, 4.0])
        mask = torch.ones(2, dtype=torch.bool)

        masker = Masker(
            mask_dict={'value': mask},
            y_hat_dict={'value': y_hat},
            y_dict={'value': y},
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, 2)
        # MSE = ((1-2)^2 + (2-4)^2) / 2 = (1 + 4) / 2 = 2.5
        self.assertAlmostEqual(loss.item(), 2.5, places=5)

    def test_compute_loss_with_policy_scaling(self):
        # Use KLDivLoss which is what WinShareActionValueHead actually uses
        model = self._make_mock_model(nn.KLDivLoss, policy_scaling=True)
        lt = BasicLossTerm(head='policy', weight=1.0)
        lt.post_init(model)

        B, A, P = 2, 3, 2  # 2 samples, 3 actions, 2 players
        # y_hat: log-probabilities (B, A, P)
        y_hat = torch.log_softmax(torch.randn(B, A, P), dim=-1)
        # y: target distributions (B, A, P) — non-negative values mark valid actions
        y = torch.zeros(B, A, P)
        y[0, :, :] = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]])
        y[1, :2, :] = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        y[1, 2, :] = -1.0  # invalid action

        mask = torch.ones(B, dtype=torch.bool)

        masker = Masker(
            mask_dict={'policy': mask},
            y_hat_dict={'policy': y_hat},
            y_dict={'policy': y},
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, B)
        self.assertTrue(torch.isfinite(loss))

    def test_non_finite_raises(self):
        model = self._make_mock_model(nn.CrossEntropyLoss, policy_scaling=True)
        lt = BasicLossTerm(head='policy', weight=1.0)
        lt.post_init(model)

        B, A = 2, 3
        y_hat = torch.tensor([[[float('inf'), 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
        y = torch.tensor([[[0.7, 0.2, 0.1]], [[0.5, 0.3, 0.2]]])
        mask = torch.ones(B, dtype=torch.bool)

        masker = Masker(
            mask_dict={'policy': mask},
            y_hat_dict={'policy': y_hat},
            y_dict={'policy': y},
        )

        with self.assertRaises(ValueError):
            lt.compute_loss(masker)

    def test_policy_scaling_zero_denominator(self):
        model = self._make_mock_model(nn.CrossEntropyLoss, policy_scaling=True)
        lt = BasicLossTerm(head='policy', weight=1.0)
        lt.post_init(model)

        B, A = 2, 3
        y_hat = torch.randn(B, 1, A)
        # All targets negative → all invalid → denominator=0
        y = torch.full((B, 1, A), -1.0)
        mask = torch.ones(B, dtype=torch.bool)

        masker = Masker(
            mask_dict={'policy': mask},
            y_hat_dict={'policy': y_hat},
            y_dict={'policy': y},
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestValueUncertaintyLossTerm(unittest.TestCase):

    def test_init_params(self):
        lt = ValueUncertaintyLossTerm(name='value_uncertainty', weight=0.5)
        self.assertEqual(lt.name, 'value_uncertainty')
        self.assertEqual(lt.weight, 0.5)
        self.assertEqual(lt._value_name, 'value')
        self.assertEqual(lt._future_mcts_value_name, 'future_mcts_value')

    def test_init_custom_params(self):
        lt = ValueUncertaintyLossTerm(
            name='vu', weight=1.0,
            value_name='my_value',
            future_mcts_value_name='my_future_mcts_value',
        )
        self.assertEqual(lt._value_name, 'my_value')
        self.assertEqual(lt._future_mcts_value_name, 'my_future_mcts_value')

    def test_compute_loss_smoke(self):
        """Verify compute_loss runs without error on valid synthetic tensors."""
        lt = ValueUncertaintyLossTerm(name='value_uncertainty', weight=1.0)

        # Mock value head with to_win_share converting 3-dim logits to 2-dim probs
        value_head = MagicMock()
        def to_win_share(logits):
            wld = logits.softmax(dim=-1)  # (B, 3)
            return wld[:, :2] + 0.5 * wld[:, 2:]
        value_head.to_win_share = to_win_share

        # Mock uncertainty head
        unc_head = MagicMock()
        unc_head.default_loss_function.return_value = nn.MSELoss

        def get_head(name):
            if name == 'value':
                return value_head
            return unc_head

        model = MagicMock()
        model.get_head.side_effect = get_head
        lt.post_init(model)

        B = 4
        n_players = 2
        # predicted_sq_delta: uncertainty predictions (B, 2)
        predicted_sq_delta = torch.rand(B, n_players)
        # value logits (B, 3) — WinLossDrawValueHead output
        lR = torch.randn(B, 3)
        # future_mcts_value targets (B, 2)
        F = torch.rand(B, n_players)

        all_mask = torch.ones(B, dtype=torch.bool)
        masker = Masker(
            mask_dict={
                'future_mcts_value': all_mask,
            },
            y_hat_dict={
                'value_uncertainty': predicted_sq_delta,
                'value': lR,
            },
            y_dict={
                'future_mcts_value': F,
            },
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, B)
        self.assertTrue(torch.isfinite(loss))


class TestBackupLossTerm(unittest.TestCase):

    @staticmethod
    def _make_post_init_model(value_dim: int = 3):
        """A MagicMock model that exposes a value head with a WLD-style to_win_share and CE
        loss, plus a value_uncertainty head with HuberLoss."""
        def to_win_share(logits):
            wld = logits.softmax(dim=-1)
            if value_dim == 3:
                return wld[:, :2] + 0.5 * wld[:, 2:]
            return wld  # WL: (B, 2) already a win-share over players

        value_head = MagicMock()
        value_head.to_win_share = to_win_share
        value_head.default_loss_function.return_value = nn.CrossEntropyLoss

        vu_head = MagicMock()
        vu_head.default_loss_function.return_value = lambda: nn.HuberLoss(delta=0.1)

        def get_head(name):
            if name == 'value':
                return value_head
            if name == 'value_uncertainty':
                return vu_head
            raise KeyError(name)

        model = MagicMock()
        model.get_head.side_effect = get_head
        return model

    def test_init_params(self):
        lt = BackupLossTerm(name='backup_net', weight=0.5,
                            q_weight=1.5, w_weight=32.0)
        self.assertEqual(lt.name, 'backup_net')
        self.assertEqual(lt.weight, 0.5)
        self.assertEqual(lt._q_weight, 1.5)
        self.assertEqual(lt._w_weight, 32.0)
        self.assertEqual(lt._value_name, 'value')
        self.assertEqual(lt._value_uncertainty_name, 'value_uncertainty')
        self.assertEqual(lt._future_mcts_value_name, 'future_mcts_value')

    def test_init_custom_params(self):
        lt = BackupLossTerm(
            name='bn', weight=1.0,
            q_weight=2.0, w_weight=3.0,
            value_name='my_value',
            value_uncertainty_name='my_vu',
            future_mcts_value_name='my_future_mcts_value',
        )
        self.assertEqual(lt._value_name, 'my_value')
        self.assertEqual(lt._value_uncertainty_name, 'my_vu')
        self.assertEqual(lt._future_mcts_value_name, 'my_future_mcts_value')

    def test_compute_loss_smoke(self):
        """Verify compute_loss runs without error on valid synthetic tensors."""
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.5, w_weight=32.0)
        lt.post_init(self._make_post_init_model(value_dim=3))
        # Disable bootstrap so this test exercises the "true target" path.
        lt.set_generation(10 ** 9)

        B = 4
        n_players = 2
        value_dim = 3
        # backup_net output (B, value_dim + 1) = [WLD logits ...; W scalar]
        backup_out = torch.randn(B, value_dim + 1)
        # value target: W/L/D one-hot game results (B, 3).
        value_target = torch.zeros(B, value_dim)
        value_target[torch.arange(B), torch.randint(0, value_dim, (B,))] = 1.0
        # future_mcts_value (B, n_players) win-shares
        F = torch.rand(B, n_players)

        all_mask = torch.ones(B, dtype=torch.bool)
        masker = Masker(
            mask_dict={
                'value': all_mask,
                'future_mcts_value': all_mask,
            },
            y_hat_dict={
                'backup_net': backup_out,
            },
            y_dict={
                'value': value_target,
                'future_mcts_value': F,
            },
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, B)
        self.assertTrue(torch.isfinite(loss))

    def test_input_mask_intersection_restricts_samples(self):
        """BackupLossTerm should only see samples where S_baseline (etc.) are valid."""
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.5, w_weight=32.0)
        lt.post_init(self._make_post_init_model(value_dim=3))
        # Disable bootstrap so this test exercises the "true target" path.
        lt.set_generation(10 ** 9)

        B = 6
        value_dim = 3
        backup_out = torch.randn(B, value_dim + 1)
        value_target = torch.zeros(B, value_dim)
        value_target[:, 0] = 1.0
        F = torch.rand(B, 2)

        all_mask = torch.ones(B, dtype=torch.bool)
        # Backup-regime samples: only first 2.
        backup_mask = torch.tensor([True, True, False, False, False, False])

        masker = Masker(
            mask_dict={
                'value': all_mask,
                'future_mcts_value': all_mask,
            },
            y_hat_dict={
                'backup_net': backup_out,
            },
            y_dict={
                'value': value_target,
                'future_mcts_value': F,
            },
            input_mask_dict={
                'value_baseline': backup_mask,
                'value_uncertainty_baseline': backup_mask,
                'child_stats': backup_mask,
            },
            input_deps={
                'backup_net': frozenset({
                    'value_baseline', 'value_uncertainty_baseline', 'child_stats',
                }),
            },
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, 2)
        self.assertTrue(torch.isfinite(loss))

    # ------------------------------------------------------------------
    # Bootstrap-anneal (match-S_baseline/Ws_baseline) tests.
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bootstrap_masker(B, backup_out, s_baseline, ws_star,
                               value_target=None, future_mcts_value=None):
        value_dim = 3
        if value_target is None:
            value_target = torch.zeros(B, value_dim)
            value_target[:, 0] = 1.0
        if future_mcts_value is None:
            future_mcts_value = torch.rand(B, 2)
        all_mask = torch.ones(B, dtype=torch.bool)
        return Masker(
            mask_dict={
                'value': all_mask,
                'future_mcts_value': all_mask,
            },
            y_hat_dict={
                'backup_net': backup_out,
            },
            y_dict={
                'value': value_target,
                'future_mcts_value': future_mcts_value,
            },
            input_mask_dict={
                'value_baseline': all_mask,
                'value_uncertainty_baseline': all_mask,
                'child_stats': all_mask,
            },
            input_deps={
                'backup_net': frozenset({
                    'value_baseline', 'value_uncertainty_baseline', 'child_stats',
                }),
            },
            input_value_dict={
                'value_baseline': s_baseline,
                'value_uncertainty_baseline': ws_star,
                'child_stats': torch.zeros(B, 1, 6),
            },
        )

    def test_set_generation_alpha_schedule(self):
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.0, w_weight=1.0,
                            bootstrap_start_gen=2, bootstrap_end_gen=12)
        lt.set_generation(0); self.assertAlmostEqual(lt._alpha, 1.0)
        lt.set_generation(2); self.assertAlmostEqual(lt._alpha, 1.0)
        lt.set_generation(7); self.assertAlmostEqual(lt._alpha, 0.5)
        lt.set_generation(12); self.assertAlmostEqual(lt._alpha, 0.0)
        lt.set_generation(99); self.assertAlmostEqual(lt._alpha, 0.0)

    def test_default_schedule_is_pure_bootstrap(self):
        """v1 default keeps the system in pure-bootstrap mode for a very long time."""
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.0, w_weight=1.0)
        lt.set_generation(0); self.assertEqual(lt._alpha, 1.0)
        lt.set_generation(10_000); self.assertGreater(lt._alpha, 0.999)

    def test_bootstrap_loss_matches_anchor_ce(self):
        """At alpha=1, the loss equals q_weight*CE(Q_logits, s_baseline) + w_weight*Huber.

        With Q_logits = log(s_baseline) (a one-hot distribution so log(0) terms drop out via
        target=0), CE collapses to the entropy of s_baseline, which is 0 for one-hot. With
        W_pred = ws_star + 1e-8, the Huber term is 0 too, so total loss is ~0.
        """
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.0, w_weight=1.0)
        lt.post_init(self._make_post_init_model(value_dim=3))
        lt.set_generation(0)  # alpha = 1

        B = 5
        # One-hot s_baseline: column 0 wins. Entropy = 0.
        s_baseline = torch.zeros(B, 3)
        s_baseline[:, 0] = 1.0
        # Q_logits: very large for the winning channel, very negative elsewhere => softmax ~ s_baseline.
        very_neg = -50.0
        Q_logits = torch.full((B, 3), very_neg)
        Q_logits[:, 0] = 0.0
        ws_star = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.3])
        # Anchor target adds +1e-8; have W_pred match exactly that for zero loss.
        W_pred = ws_star + 1e-8
        backup_out = torch.cat([Q_logits, W_pred.unsqueeze(1)], dim=1)

        masker = self._make_bootstrap_masker(B, backup_out, s_baseline, ws_star)
        loss, n = lt.compute_loss(masker)
        self.assertEqual(n, B)
        self.assertLess(float(loss), 1e-6)

    def test_bootstrap_loss_positive_when_outputs_mismatch_anchors(self):
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.0, w_weight=1.0)
        lt.post_init(self._make_post_init_model(value_dim=3))
        lt.set_generation(0)  # alpha = 1

        B = 4
        backup_out = torch.zeros(B, 4)  # uniform Q (softmax = 1/3), zero W
        # Highly peaked s_baseline, very different from uniform => positive CE.
        s_baseline = torch.zeros(B, 3)
        s_baseline[:, 0] = 1.0
        ws_star = torch.full((B,), 0.1)
        masker = self._make_bootstrap_masker(B, backup_out, s_baseline, ws_star)
        loss, _ = lt.compute_loss(masker)
        self.assertGreater(float(loss), 1e-3)

    def test_bootstrap_path_ignores_true_targets(self):
        """At alpha=1, value_target / future_mcts_value should not affect the loss."""
        lt = BackupLossTerm(name='backup_net', weight=1.0,
                            q_weight=1.0, w_weight=1.0)
        lt.post_init(self._make_post_init_model(value_dim=3))
        lt.set_generation(0)  # alpha = 1

        B = 3
        s_baseline = torch.zeros(B, 3)
        s_baseline[:, 0] = 1.0
        very_neg = -50.0
        Q_logits = torch.full((B, 3), very_neg)
        Q_logits[:, 0] = 0.0
        ws_star = torch.full((B,), -1e-8)
        W_pred = torch.zeros(B)
        backup_out = torch.cat([Q_logits, W_pred.unsqueeze(1)], dim=1)

        # Vary the "true" targets between two maskers; loss should be unchanged.
        vt_a = torch.zeros(B, 3); vt_a[:, 0] = 1.0
        vt_b = torch.zeros(B, 3); vt_b[:, 2] = 1.0
        F_a = torch.zeros(B, 2)
        F_b = torch.ones(B, 2)
        masker_a = self._make_bootstrap_masker(B, backup_out, s_baseline, ws_star, vt_a, F_a)
        masker_b = self._make_bootstrap_masker(B, backup_out, s_baseline, ws_star, vt_b, F_b)
        loss_a, _ = lt.compute_loss(masker_a)
        loss_b, _ = lt.compute_loss(masker_b)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=6)
        self.assertLess(float(loss_a), 1e-6)


if __name__ == '__main__':
    unittest.main()
