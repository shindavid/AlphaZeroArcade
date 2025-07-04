from alphazero.logic.agent_types import Agent, AgentDBId, AgentRole, IndexedAgent
from alphazero.logic.arena import RatingData
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.match_runner import MatchType
from alphazero.logic.ratings import estimate_elo_newton, win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

import numpy as np

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


@dataclass
class EvalRatingData:
    evaluated_iagents: List[IndexedAgent]
    ratings: np.ndarray
    tag: str
    lookup_elo: Dict[int, float] = field(init=False)

    def __post_init__(self):
        self.lookup_elo = {iagent.index: rating for iagent, rating in zip(self.evaluated_iagents, self.ratings)}

    def update(self, iagent: IndexedAgent, rating: float):
        self.evaluated_iagents.append(iagent)
        self.ratings = np.concatenate((self.ratings, [rating]))
        self.lookup_elo[iagent.index] = rating

class EvalUtils:
    @staticmethod
    def estimate_rating_nearby_gens(gen: int, evaluated_gens: List[int], ratings: List[float]) -> float:
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
    def gen_matches(estimated_rating: float, ixs: List[int], elos: List[float], n_games: int, top_k: int=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        The opponent selection algorithm is adapted from KataGo:
        "Accelerating Self-Play Learning in Go" by David J. Wu (Section 5.1).

        The match generating process follows these steps:

        1. Compute the test agent's probability of winning against each committee member on the
           difference in their estimated Elo ratings. By default, the initial estimated rating of
           the test agent is set to be the mean of the committee members' ratings if it has not
           played any matches yet. The initial estimate can be provided by the caller.
        2. Calculate the variance of the win probability for each committee member and peer test agents using
           p * (1 - p), where p is the win probability.
        3. Select opponents in proportion to their win probability variance.
        4. Remove any opponents that the test agent has already played from the sampling pool.
        5. Pick the top k opponents with the highest number of matches played and redistribute the
           remaining matches among them. The redistribution is done by calculating the percentage of
           matches played by each opponent and multiplying it by the number of remaining matches.
        6. After each match, update the test agent's estimated rating. If the new rating
           deviates beyond `error_threshold` from the original estimate, it indicates that
           the initial estimate was unreliable. In this case, reset the process and return to step 1.
        7. If the test agent has played against all committee members or has completed
           a sufficient number of matches, stop further evaluation.
        8. Compute the final rating by interpolating from the benchmark committee's ratings
           before any games were played against the test agent.
        """

        ixs = np.array(ixs, dtype=int)

        logger.debug(f"estimated_rating: {estimated_rating}")
        logger.debug(f"potential_opponent_ixs: {ixs}")
        logger.debug(f"elo ratings: {elos}")

        p = [win_prob(estimated_rating, elo) for elo in elos]
        var = np.array([q * (1 - q) for q in p])
        var = var / np.sum(var)

        logger.debug(f"winning probs: {list(zip(ixs, p))}")
        logger.debug(f"vars: {list(zip(ixs, var))}")

        sample_ixs = ixs[np.random.choice(len(ixs), p=var, size=n_games)]
        chosen_ixs, num_matches = np.unique(sample_ixs, return_counts=True)

        if len(chosen_ixs) > top_k:
            top_k_ixs = np.argsort(num_matches)[-top_k:]
            top_chosen_ixs = chosen_ixs[top_k_ixs]
            top_num_matches = num_matches[top_k_ixs]
            top_match_percent = top_num_matches / np.sum(top_num_matches)
            tail_matches = np.sum(num_matches) - np.sum(top_num_matches)
            redistrubution = top_match_percent * tail_matches
            num_matches = np.round(top_num_matches + redistrubution).astype(int)
            rounding_error = np.sum(num_matches) - n_games
            if rounding_error > 0:
                num_matches[-1] -= rounding_error
            elif rounding_error < 0:
                num_matches[-1] += abs(rounding_error)
            chosen_ixs = top_chosen_ixs

        return chosen_ixs, num_matches
