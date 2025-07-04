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
from typing import Dict, List, Optional, Set


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

class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_tag: str):
        self._organizer = organizer
        benchmark = Benchmarker(organizer, db_filename=organizer.eval_db_filename(benchmark_tag))
        self._benchmark_rating_data = benchmark.read_ratings_from_db()
        self._arena = benchmark.clone_arena()
        self.db = RatingDB(self._organizer.eval_db_filename(benchmark_tag))
        self.load_from_db()

    def load_from_db(self):
        self._arena.load_agents_from_db(self.db, role=AgentRole.TEST)
        self._arena.load_matches_from_db(self.db, type=MatchType.EVALUATE)
        self._elo_ratings = self.read_ratings_from_db()

    def gen_matches(self, estimated_rating: float, opponent_ix_played: np.ndarray, n_games: int, top_k: int=5):
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

        potential_opponent_ixs = np.where(self.benchmark_committee)[0]

        logger.debug(f"estimated_rating: {estimated_rating}")
        logger.debug(f"potential_opponent_ixs: {potential_opponent_ixs}")
        logger.debug(f"arena ratings: {self.benchmark_ratings}")

        p = [win_prob(estimated_rating, self.benchmark_ratings[ix]) for ix in potential_opponent_ixs]
        var = np.array([q * (1 - q) for q in p])
        mask = np.zeros(len(var), dtype=bool)
        potential_opponents_played = np.where(np.isin(potential_opponent_ixs, opponent_ix_played))[0]
        mask[potential_opponents_played] = True
        var[mask] = 0
        var = var / np.sum(var)

        logger.debug(f"winning probs: {list(zip(potential_opponent_ixs, p))}")
        logger.debug(f"vars: {list(zip(potential_opponent_ixs, var))}")

        sample_ixs = potential_opponent_ixs[np.random.choice(len(potential_opponent_ixs), p=var, size=n_games)]
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

    def eval_elo(self, ix: int) -> float:
        if ix < 0 or ix >= len(self._arena.indexed_agents):
            raise IndexError(f"Invalid agent index: {ix}")

        n_games_played = self._arena._W_matrix[ix, :] + self._arena._W_matrix[:, ix]
        n_games_won = self._arena._W_matrix[ix, :]

        played_ixs = np.where(n_games_played > 0)[0]
        n = n_games_played[played_ixs]
        k = n_games_won[played_ixs]
        elos = np.array([self._benchmark_rating_data.lookup_elo[i] for i in played_ixs])
        min_elo = self.benchmark_ratings.min()
        max_elo = self.benchmark_ratings.max()
        return estimate_elo_newton(n, k, elos, lower=min_elo, upper=max_elo)

    def test_agent_ixs(self) -> np.ndarray:
        test_ixs = [iagent.index for iagent in self._arena.indexed_agents if AgentRole.TEST in iagent.roles]
        return np.array(test_ixs)

    def benchmark_agent_ixs(self) -> np.ndarray:
        benchmark_ixs = [iagent.index for iagent in self._arena.indexed_agents if AgentRole.BENCHMARK in iagent.roles]
        return np.array(benchmark_ixs)

    def read_ratings_from_db(self) -> EvalRatingData:
        rating_data: RatingData = self._arena.load_ratings_from_db(self.db, AgentRole.TEST)
        ratings = rating_data.ratings
        evaluated_iagents = [self._arena.agent_lookup_db_id[db_id] for db_id in rating_data.agent_ids]
        return EvalRatingData(evaluated_iagents, ratings, rating_data.tag)

    def add_agent(self, agent: Agent, roles: Set[AgentRole], expand_matrix: bool=True, db: Optional[RatingDB]=None):
        return self._arena.add_agent(agent, roles, expand_matrix=expand_matrix, db=db)

    @property
    def benchmark_ratings(self) -> np.ndarray:
        return self._benchmark_rating_data.ratings

    @property
    def benchmark_committee(self) -> BenchmarkRatingData:
        return self._benchmark_rating_data.committee

    @property
    def arena_ratings(self) -> np.ndarray:
        return self._arena.ratings

    @property
    def indexed_agents(self) -> List[Agent]:
        return self._arena.indexed_agents

    @property
    def agent_lookup(self) -> Dict[Agent, IndexedAgent]:
        return self._arena._agent_lookup

    @property
    def agent_lookup_db_id(self) -> Dict[AgentDBId, IndexedAgent]:
        return self._arena._agent_lookup_db_id


class EvalUtils:
    @staticmethod
    def estimate_rating_nearby_gens(gen: int, evaluated_gens: List[int], ratings: np.ndarray) -> float:
        assert len(evaluated_gens) == len(ratings)
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
