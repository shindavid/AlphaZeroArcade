from alphazero.logic.benchmarker import Benchmarker
from util.index_set import IndexSet

import numpy as np
import unittest

class TestSelectCommittee(unittest.TestCase):
    def test_simple_case(self):
        elos = np.array([1000, 1300, 1400])
        min_elo_gap = 200
        committee = Benchmarker.select_committee(elos, min_elo_gap)
        self.assertTrue(np.array_equal(committee, [1, 0, 1]))

    def test_elo_len_1(self):
        elos = np.array([1000])
        min_elo_gap = 200
        committee = Benchmarker.select_committee(elos, min_elo_gap)
        self.assertTrue(np.array_equal(committee, [1]))

    def test_not_sorted_elos(self):
        elos = np.array([1400, 1000, 1300])
        min_elo_gap = 200
        committee = Benchmarker.select_committee(elos, min_elo_gap)
        self.assertTrue(np.array_equal(committee, [1, 1, 0]))

    def test_duplicate_elos(self):
        elos = np.array([1000, 1300, 1300, 1300])
        min_elo_gap = 200
        committee = Benchmarker.select_committee(elos, min_elo_gap)
        self.assertTrue(np.array_equal(committee, [1, 0, 0, 1]))

if __name__ == '__main__':
    unittest.main()