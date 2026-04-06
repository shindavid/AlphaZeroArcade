from shared.rating_params import (
    DefaultTargetEloGap,
    RatingPlayerOptions,
    RatingParams,
)

import unittest


class TestDefaultTargetEloGap(unittest.TestCase):

    def test_default_values(self):
        gap = DefaultTargetEloGap()
        self.assertEqual(gap.first_run, 500.0)
        self.assertEqual(gap.benchmark, 100.0)

    def test_custom_values(self):
        gap = DefaultTargetEloGap(first_run=300.0, benchmark=50.0)
        self.assertEqual(gap.first_run, 300.0)
        self.assertEqual(gap.benchmark, 50.0)

    def test_negative_first_run_raises(self):
        with self.assertRaises(AssertionError):
            DefaultTargetEloGap(first_run=-1.0)

    def test_negative_benchmark_raises(self):
        with self.assertRaises(AssertionError):
            DefaultTargetEloGap(benchmark=-1.0)

    def test_zero_first_run_raises(self):
        with self.assertRaises(AssertionError):
            DefaultTargetEloGap(first_run=0.0)

    def test_first_run_less_than_benchmark_raises(self):
        with self.assertRaises(AssertionError):
            DefaultTargetEloGap(first_run=50.0, benchmark=100.0)

    def test_equal_values_ok(self):
        gap = DefaultTargetEloGap(first_run=100.0, benchmark=100.0)
        self.assertEqual(gap.first_run, gap.benchmark)


class TestRatingPlayerOptions(unittest.TestCase):

    def test_default_values(self):
        opts = RatingPlayerOptions()
        self.assertEqual(opts.num_search_threads, 4)
        self.assertEqual(opts.num_iterations, 100)

    def test_custom_values(self):
        opts = RatingPlayerOptions(num_search_threads=8, num_iterations=200)
        self.assertEqual(opts.num_search_threads, 8)
        self.assertEqual(opts.num_iterations, 200)

    def test_zero_threads_raises(self):
        with self.assertRaises(AssertionError):
            RatingPlayerOptions(num_search_threads=0)

    def test_negative_iterations_raises(self):
        with self.assertRaises(AssertionError):
            RatingPlayerOptions(num_iterations=-1)


class TestRatingParams(unittest.TestCase):

    def test_default_values(self):
        rp = RatingParams()
        self.assertEqual(rp.eval_error_threshold, 50.0)
        self.assertEqual(rp.n_games_per_self_evaluation, 100)
        self.assertEqual(rp.n_games_per_evaluation, 1000)
        self.assertIsNone(rp.target_elo_gap)
        self.assertFalse(rp.use_remote_play)
        self.assertEqual(rp.rating_tag, '')

    def test_zero_games_per_evaluation_raises(self):
        with self.assertRaises(AssertionError):
            RatingParams(n_games_per_evaluation=0)

    def test_zero_games_per_self_evaluation_raises(self):
        with self.assertRaises(AssertionError):
            RatingParams(n_games_per_self_evaluation=0)

    def test_negative_target_elo_gap_raises(self):
        with self.assertRaises(AssertionError):
            RatingParams(target_elo_gap=-10.0)

    def test_zero_target_elo_gap_raises(self):
        with self.assertRaises(AssertionError):
            RatingParams(target_elo_gap=0.0)

    def test_valid_target_elo_gap(self):
        rp = RatingParams(target_elo_gap=200.0)
        self.assertEqual(rp.target_elo_gap, 200.0)

    def test_add_to_cmd_loop_controller(self):
        rp = RatingParams(
            rating_player_options=RatingPlayerOptions(num_iterations=200),
        )
        cmd = []
        rp.add_to_cmd(cmd, loop_controller=True)
        self.assertIn('--num-iterations', cmd)
        self.assertIn('200', cmd)

    def test_add_to_cmd_server(self):
        rp = RatingParams(
            rating_player_options=RatingPlayerOptions(num_search_threads=8),
            use_remote_play=True,
            rating_tag='test',
        )
        cmd = []
        rp.add_to_cmd(cmd, server=True)
        self.assertIn('--num-search-threads', cmd)
        self.assertIn('--use-remote-play', cmd)
        self.assertIn('--rating-tag', cmd)
        self.assertIn('test', cmd)

    def test_add_to_cmd_defaults_empty(self):
        rp = RatingParams()
        cmd = []
        rp.add_to_cmd(cmd, loop_controller=True, server=True)
        self.assertEqual(cmd, [])


if __name__ == '__main__':
    unittest.main()
