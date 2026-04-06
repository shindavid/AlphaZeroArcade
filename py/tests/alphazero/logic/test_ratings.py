import unittest
import numpy as np
from alphazero.logic.ratings import (
    BETA_SCALE_FACTOR,
    WinLossDrawCounts,
    MatchRecord,
    extract_match_record,
    compute_ratings,
    win_prob,
)


class TestWinProb(unittest.TestCase):
    def test_equal_ratings_is_half(self):
        self.assertAlmostEqual(win_prob(0, 0), 0.5)

    def test_100_point_advantage_gives_64_percent(self):
        # By definition of BETA_SCALE_FACTOR
        self.assertAlmostEqual(win_prob(100, 0), 0.64, places=5)

    def test_symmetry(self):
        self.assertAlmostEqual(win_prob(200, 100) + win_prob(100, 200), 1.0)

    def test_higher_elo_wins_more(self):
        self.assertGreater(win_prob(500, 0), win_prob(100, 0))

    def test_numpy_array_input(self):
        elos = np.array([0.0, 100.0, 200.0])
        result = win_prob(300.0, elos)
        self.assertEqual(result.shape, (3,))
        # Higher opponent elo => lower win prob for us at 300
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])


class TestWinLossDrawCounts(unittest.TestCase):
    def test_all_wins(self):
        self.assertAlmostEqual(WinLossDrawCounts(10, 0, 0).win_rate(), 1.0)

    def test_all_losses(self):
        self.assertAlmostEqual(WinLossDrawCounts(0, 10, 0).win_rate(), 0.0)

    def test_all_draws(self):
        self.assertAlmostEqual(WinLossDrawCounts(0, 0, 10).win_rate(), 0.5)

    def test_zero_games(self):
        self.assertEqual(WinLossDrawCounts(0, 0, 0).win_rate(), 0)

    def test_mixed(self):
        # 2W 2L 2D: (2*2 + 2) / (2*6) = 6/12 = 0.5
        self.assertAlmostEqual(WinLossDrawCounts(2, 2, 2).win_rate(), 0.5)

    def test_n_games(self):
        self.assertEqual(WinLossDrawCounts(3, 4, 5).n_games, 12)

    def test_json_round_trip(self):
        orig = WinLossDrawCounts(7, 3, 2)
        restored = WinLossDrawCounts.from_json(orig.to_json())
        self.assertEqual(restored.win, orig.win)
        self.assertEqual(restored.loss, orig.loss)
        self.assertEqual(restored.draw, orig.draw)

    def test_iadd(self):
        a = WinLossDrawCounts(1, 2, 3)
        a += WinLossDrawCounts(4, 5, 6)
        self.assertEqual((a.win, a.loss, a.draw), (5, 7, 9))


class TestMatchRecord(unittest.TestCase):
    def test_update_and_get(self):
        record = MatchRecord()
        record.update(0, WinLossDrawCounts(5, 3, 2))
        counts = record.get(0)
        self.assertEqual(counts.win, 5)
        self.assertEqual(counts.loss, 3)
        self.assertEqual(counts.draw, 2)

    def test_update_accumulates(self):
        record = MatchRecord()
        record.update(0, WinLossDrawCounts(1, 0, 0))
        record.update(0, WinLossDrawCounts(2, 1, 0))
        counts = record.get(0)
        self.assertEqual(counts.win, 3)
        self.assertEqual(counts.loss, 1)

    def test_json_round_trip(self):
        record = MatchRecord()
        record.update(0, WinLossDrawCounts(51, 49, 0))
        record.update(1, WinLossDrawCounts(49, 51, 0))
        restored = MatchRecord.from_json(record.to_json())
        self.assertEqual(restored.get(0).win, 51)
        self.assertEqual(restored.get(1).win, 49)

    def test_empty(self):
        self.assertTrue(MatchRecord().empty())

    def test_not_empty_after_update(self):
        record = MatchRecord()
        record.update(0, WinLossDrawCounts(1, 0, 0))
        self.assertFalse(record.empty())


class TestExtractMatchRecord(unittest.TestCase):
    NO_TIMESTAMP_LINES = [
        'All games complete!',
        'pid=0 name=Perfect-21 W51 L49 D0 [51]',
        'pid=1 name=alpha0-C-100 W49 L51 D0 [49]',
    ]

    TIMESTAMP_LINES = [
        '2024-03-13 14:10:09.131334 All games complete!',
        '2024-03-13 14:10:09.131368 pid=0 name=Perfect-21 W51 L49 D0 [51]',
        '2024-03-13 14:10:09.131382 pid=1 name=alpha0-C-100 W49 L51 D0 [49]',
    ]

    def test_no_timestamp_format(self):
        record = extract_match_record(self.NO_TIMESTAMP_LINES)
        self.assertEqual(record.get(0).win, 51)
        self.assertEqual(record.get(0).loss, 49)
        self.assertEqual(record.get(1).win, 49)
        self.assertEqual(record.get(1).loss, 51)

    def test_timestamp_format(self):
        record = extract_match_record(self.TIMESTAMP_LINES)
        self.assertEqual(record.get(0).win, 51)
        self.assertEqual(record.get(1).win, 49)

    def test_string_input(self):
        stdout = '\n'.join(self.NO_TIMESTAMP_LINES)
        record = extract_match_record(stdout)
        self.assertEqual(record.get(0).win, 51)

    def test_symmetry_enforced(self):
        # wins for pid=0 == losses for pid=1 and vice versa
        record = extract_match_record(self.NO_TIMESTAMP_LINES)
        c0 = record.get(0)
        c1 = record.get(1)
        self.assertEqual(c0.win, c1.loss)
        self.assertEqual(c0.loss, c1.win)
        self.assertEqual(c0.draw, c1.draw)

    def test_includes_draws(self):
        lines = [
            'pid=0 name=A W40 L40 D20 [60]',
            'pid=1 name=B W40 L40 D20 [60]',
        ]
        record = extract_match_record(lines)
        self.assertEqual(record.get(0).draw, 20)


class TestComputeRatings(unittest.TestCase):
    def test_equal_play_gives_equal_ratings(self):
        w = np.array([[0, 5], [5, 0]], dtype=float)
        beta = compute_ratings(w)
        # Both should be close to 0 (pinned at player 0's rating)
        self.assertAlmostEqual(beta[0], 0.0, places=4)
        self.assertAlmostEqual(beta[1], 0.0, places=4)

    def test_player_zero_wins_more_means_player_one_lower(self):
        w = np.array([[0, 8], [2, 0]], dtype=float)
        beta = compute_ratings(w)
        self.assertEqual(beta[0], 0.0)
        self.assertLess(beta[1], 0.0)

    def test_three_players_ordering(self):
        # Player 0 > player 1 > player 2
        w = np.array([
            [0, 8, 9],
            [2, 0, 8],
            [1, 2, 0],
        ], dtype=float)
        beta = compute_ratings(w)
        self.assertEqual(beta[0], 0.0)
        self.assertLess(beta[1], beta[0])
        self.assertLess(beta[2], beta[1])

    def test_eps_handles_disconnected(self):
        # Player 0 never plays player 1 — eps keeps it well-defined
        w = np.array([[0, 0], [0, 0]], dtype=float)
        beta = compute_ratings(w, eps=1.0)
        self.assertAlmostEqual(beta[0], 0.0, places=4)
        self.assertAlmostEqual(beta[1], 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
