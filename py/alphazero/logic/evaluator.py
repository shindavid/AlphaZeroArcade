from alphazero.logic.ratings import win_prob
import numpy as np

import logging
from typing import Dict, List


logger = logging.getLogger(__name__)


class EvalUtils:
    @staticmethod
    def estimate_rating_nearby_gens(gen: int, evaluated_gens: List[int], ratings: List[float])\
            -> float:
        assert len(evaluated_gens) == len(ratings)
        ratings = np.array(ratings, dtype=float)
        evaluated_gens_arr = np.array(evaluated_gens)
        sorted_ixs = np.argsort(evaluated_gens_arr)

        for i in range(len(evaluated_gens_arr) - 1):
            left_gen = evaluated_gens_arr[sorted_ixs[i]]
            right_gen = evaluated_gens_arr[sorted_ixs[i + 1]]
            if left_gen < gen < right_gen:
                left_rating = ratings[sorted_ixs[i]]
                right_rating = ratings[sorted_ixs[i + 1]]
                return np.interp(gen, [left_gen, right_gen], [left_rating, right_rating])

        if gen > evaluated_gens_arr[sorted_ixs[-1]]:
            return ratings[sorted_ixs[-1]]
        return None

    @staticmethod
    def get_next_gen_to_eval(latest_gen: int, evaluated_gens: List[int]):
        if 1 not in evaluated_gens:
            return 1
        if latest_gen not in evaluated_gens:
            return latest_gen

        left_gen, right_gen = EvalUtils.get_biggest_gen_gap(evaluated_gens)
        if left_gen + 1 < right_gen:
            gen = (left_gen + right_gen) // 2
            assert gen not in evaluated_gens
        return int(gen)

    @staticmethod
    def get_biggest_gen_gap(evaluated_gens: List[int]):
        gens = evaluated_gens.copy()
        gens = np.sort(gens)
        gaps = np.diff(gens)
        max_gap_ix = np.argmax(gaps)
        left_gen = gens[max_gap_ix]
        right_gen = gens[max_gap_ix + 1]
        return left_gen, right_gen

    @staticmethod
    def gen_matches(estimated_rating: float, ixs: List[int], elos: List[float], n_games: int,
                    top_k: int = 5) -> Dict[int, int]:
        """
        The opponent selection algorithm is adapted from KataGo:
        "Accelerating Self-Play Learning in Go" by David J. Wu (Section 5.1).

        The match generating process follows these steps:

        1. Compute the test agent's win probability against each committee member.
        2. Convert each win probability into a variance value.
        3. Select opponents in proportion to variance, zeroing out all but the top top_k opponents.
        """
        ixs = np.array(ixs, dtype=int)
        elos = np.array(elos, dtype=float)
        p = win_prob(estimated_rating, elos)
        var = p * (1 - p)
        var[np.argsort(var)[:-top_k]] = 0.0
        var /= np.sum(var)
        sample_ixs = ixs[np.random.choice(len(ixs), p=var, size=n_games)]

        num_matches = {}
        for ix, count in zip(*np.unique(sample_ixs, return_counts=True)):
            num_matches[ix] = count
        return num_matches
