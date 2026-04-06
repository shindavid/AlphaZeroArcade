from alphazero.logic.eval_vs_benchmark_utils import EvalVsBenchmarkUtils

import numpy as np
import unittest


class TestEvalUtils(unittest.TestCase):
    def test_gen_matches(self):
        np.random.seed(42)

        est_rating = 1150.0
        ixs = list(range(10))
        elos = np.array(ixs) * 200
        n_games = 10000
        top_k = 5

        num_matches = EvalVsBenchmarkUtils.gen_matches(est_rating, ixs, elos, n_games, top_k)
        pcts = {int(k): float(v / n_games) for k, v in num_matches.items()}

        self.assertAlmostEqual(pcts.get(4, -1), 0.13, delta=0.05)
        self.assertAlmostEqual(pcts.get(5, -1), 0.27, delta=0.05)
        self.assertAlmostEqual(pcts.get(6, -1), 0.31, delta=0.05)
        self.assertAlmostEqual(pcts.get(7, -1), 0.19, delta=0.05)
        self.assertAlmostEqual(pcts.get(8, -1), 0.08, delta=0.05)


if __name__ == '__main__':
    unittest.main()
