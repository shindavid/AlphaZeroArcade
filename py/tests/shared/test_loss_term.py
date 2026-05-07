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

    def test_init_params(self):
        lt = BackupLossTerm(name='backup_net', weight=0.5)
        self.assertEqual(lt.name, 'backup_net')
        self.assertEqual(lt.weight, 0.5)
        self.assertEqual(lt._value_name, 'value')
        self.assertEqual(lt._future_mcts_value_name, 'future_mcts_value')

    def test_init_custom_params(self):
        lt = BackupLossTerm(
            name='bn', weight=1.0,
            value_name='my_value',
            future_mcts_value_name='my_future_mcts_value',
        )
        self.assertEqual(lt._value_name, 'my_value')
        self.assertEqual(lt._future_mcts_value_name, 'my_future_mcts_value')

    def test_compute_loss_smoke(self):
        """Verify compute_loss runs without error on valid synthetic tensors."""
        lt = BackupLossTerm(name='backup_net', weight=1.0)
        lt.post_init(MagicMock())  # post_init is a no-op for BackupLossTerm

        B = 4
        n_players = 2
        # backup_net output (B, 2) -- [Q_pred, W_pred]
        backup_out = torch.randn(B, 2)
        # value target: W/L/D probs (B, 3), simulated as one-hot game results.
        value_target = torch.zeros(B, 3)
        value_target[torch.arange(B), torch.randint(0, 3, (B,))] = 1.0
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
        """BackupLossTerm should only see samples where Qs_star (etc.) are valid."""
        lt = BackupLossTerm(name='backup_net', weight=1.0)
        lt.post_init(MagicMock())

        B = 6
        backup_out = torch.randn(B, 2)
        value_target = torch.zeros(B, 3)
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
                'Qs_star': backup_mask,
                'Ws_star': backup_mask,
                'child_stats': backup_mask,
            },
            input_deps={
                'backup_net': frozenset({
                    'Qs_star', 'Ws_star', 'child_stats',
                }),
            },
        )

        loss, n_samples = lt.compute_loss(masker)
        self.assertEqual(n_samples, 2)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
