from alphazero.logic.agent_types import Agent, AgentRole, IndexedAgent, MCTSAgent
from alphazero.logic.arena import RatingData
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.ratings import win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

import numpy as np
from scipy.interpolate import interp1d

import logging
from dataclasses import dataclass
from typing import List, Optional


logger = logging.getLogger(__name__)


@dataclass
class EvalRatingData:
    evaluated_iagents: List[IndexedAgent]
    ratings: np.ndarray
    tag: str


class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_tag: str):
        self._organizer = organizer
        self._benchmark = Benchmarker(organizer, db_filename=organizer.eval_db_filename(benchmark_tag))
        self._benchmark_rating_data = self._benchmark.read_ratings_from_db()

        self._arena = self._benchmark.clone_arena()
        self.db = RatingDB(self._organizer.eval_db_filename(benchmark_tag))
        self.load_from_db()
        self.refresh_ratings()

    def load_from_db(self):
        self._arena.load_agents_from_db(self.db, role=AgentRole.TEST)
        self._arena.load_matches_from_db(self.db, type=MatchType.EVALUATE)

    def eval_agent(self, test_agent: Agent, n_games, error_threshold=100,
                   init_rating_estimate: Optional[float]=None):
        """
        Generic evaluation function for all types of agents.

        The opponent selection algorithm is adapted from KataGo:
        "Accelerating Self-Play Learning in Go" by David J. Wu (Section 5.1).

        The evaluation process follows these steps:

        1. Compute the test agent's probability of winning against each committee member
           based on the difference in their estimated Elo ratings. By default, the initial estimated rating of
           the test agent is set to be the mean of the committee members' ratings if it has not
           played any matches yet. The initial estimate can be provided by the caller. In the MCTSEvaluator,
           the initial estimate is interpolated using the near by gens' ratings if available.
        2. Calculate the variance of the win probability for each committee member using
           p * (1 - p), where p is the win probability.
        3. Select opponents from the committee in proportion to their win probability variance.
        4. Remove any opponents that the test agent has already played from the sampling pool.
        5. After each match, update the test agent's estimated rating. If the new rating
           deviates beyond `error_threshold` from the original estimate, it indicates that
           the initial estimate was unreliable. In this case, reset the process and return to step 1.
        6. If the test agent has played against all committee members or has completed
           a sufficient number of matches, stop further evaluation.
        7. Compute the final rating by interpolating from the benchmark committee's ratings
           before any games were played against the test agent.
        """
        self._arena.refresh_ratings()
        if init_rating_estimate is not None:
            estimated_rating = init_rating_estimate
        else:
            estimated_rating = np.mean(self.benchmark_ratings)

        test_iagent = self._arena._add_agent(test_agent, AgentRole.TEST, expand_matrix=True, db=self.db)

        n_games_played = self._arena.n_games_played(test_agent)
        if n_games_played > 0:
            n_games -= n_games_played
            estimated_rating = self._arena.ratings[test_iagent.index]

        committee_ixs = np.where(self.benchmark_committee)[0]
        opponent_ix_played = self._arena.get_past_opponents_ix(test_agent)
        while n_games > 0 and len(opponent_ix_played) < len(committee_ixs):
            opponent_ix_played = self._arena.get_past_opponents_ix(test_agent)
            chosen_ixs, num_matches = self.gen_matches(estimated_rating, opponent_ix_played, n_games)
            sorted_ixs = np.argsort(num_matches)[::-1]
            logger.debug('evaluating %s against %d opponents. Estimated rating: %f', test_agent, len(chosen_ixs), estimated_rating)
            for i in range(len(chosen_ixs)):
                ix = chosen_ixs[sorted_ixs[i]]
                n = num_matches[sorted_ixs[i]]

                opponent = self._arena.indexed_agents[ix].agent
                match = Match(test_agent, opponent, n, MatchType.EVALUATE)
                self._arena.play_matches([match], self._organizer.game, db=self.db)
                n_games -= n
                opponent_ix_played = np.concatenate([opponent_ix_played, [ix]])
                self._arena.refresh_ratings()
                new_estimate = self._arena.ratings[test_iagent.index]
                if abs(new_estimate - estimated_rating) > error_threshold:
                    estimated_rating = new_estimate
                    break

        _, interpolated_ratings = self.interpolate_ratings()
        test_iagents = [ia for ia in self._arena.indexed_agents if ia.role == AgentRole.TEST]
        self.db.commit_ratings(test_iagents, interpolated_ratings)
        logger.debug('Finished evaluating %s. Interpolated rating: %f. Before interp: %f',
                    test_agent, interpolated_ratings[-1], self.arena_ratings[-1])

    def gen_matches(self, estimated_rating: float, opponent_ix_played: np.ndarray, n_games: int):
        committee_ixs = np.where(self.benchmark_committee)[0]
        p = [win_prob(estimated_rating, self._arena.ratings[ix]) for ix in committee_ixs]
        var = np.array([q * (1 - q) for q in p])
        mask = np.zeros(len(var), dtype=bool)
        committee_ix_played = np.where(np.isin(committee_ixs, opponent_ix_played))[0]
        mask[committee_ix_played] = True
        var[mask] = 0
        var = var / np.sum(var)

        sample_ixs = committee_ixs[np.random.choice(len(committee_ixs), p=var, size=n_games)]
        chosen_ixs, num_matches = np.unique(sample_ixs, return_counts=True)
        return chosen_ixs, num_matches

    def interpolate_ratings(self) -> np.ndarray:
        benchmark_ixs = self.benchmark_agent_ixs()
        n_benchmark_ixs = len(benchmark_ixs)
        test_ixs = self.test_agent_ixs()

        self._arena.refresh_ratings()
        xs = self._arena.ratings[benchmark_ixs][self.benchmark_committee.resize(n_benchmark_ixs)]
        ys = self.benchmark_ratings[self.benchmark_committee]
        test_agents_elo = self._arena.ratings[test_ixs]
        sorted_ixs = np.argsort(xs)
        xs_sorted = xs[sorted_ixs]
        ys_sorted = ys[sorted_ixs]
        interp_func = interp1d(xs_sorted, ys_sorted, kind="linear", fill_value=(min(ys), max(ys)), bounds_error=False)
        interpolated_ratings = interp_func(test_agents_elo)

        return test_ixs, interpolated_ratings

    def test_agent_ixs(self) -> np.ndarray:
        test_ixs = [iagent.index for iagent in self._arena.indexed_agents if iagent.role == AgentRole.TEST]
        return np.array(test_ixs)

    def benchmark_agent_ixs(self) -> np.ndarray:
        benchmark_ixs = [iagent.index for iagent in self._arena.indexed_agents if iagent.role == AgentRole.BENCHMARK]
        return np.array(benchmark_ixs)

    def read_ratings_from_db(self) -> EvalRatingData:
        rating_data: RatingData = self._arena.load_ratings_from_db(self.db, AgentRole.TEST)
        ratings = rating_data.ratings
        evaluated_iagents = [self._arena.agent_lookup_db_id[db_id] for db_id in rating_data.agent_ids]
        return EvalRatingData(evaluated_iagents, ratings, rating_data.tag)

    def refresh_ratings(self):
        self._arena.refresh_ratings()

    def add_agent(self, agent: Agent, role: AgentRole, expand_matrix: bool=True, db: Optional[RatingDB]=None):
        return self._arena._add_agent(agent, role, expand_matrix=expand_matrix, db=db)

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
    def agent_lookup(self) -> dict:
        return self._arena._agent_lookup


class MCTSEvaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_tag: str):
        self._organizer = organizer
        self._evaluator = Evaluator(organizer, benchmark_tag)
        rating_data = self._evaluator.read_ratings_from_db()
        self._evaluated_ixs = [self._evaluator.agent_lookup[agent].index \
            for agent in rating_data.evaluated_agents]

    def run(self, n_iters: int=100, target_eval_percent: float=1.0, n_games: int=100, error_threshold=100):
        self._evaluator.refresh_ratings()
        while True:
            evaluated_gens = [self._evaluator.indexed_agents[ix].agent.gen \
                for ix in self._evaluated_ixs]
            last_gen = self._organizer.get_latest_model_generation()

            evaluated_percent = len(evaluated_gens) / (last_gen + 1)
            if evaluated_percent >= target_eval_percent:
                break
            gen = EvalUtils.get_next_gen_to_eval(last_gen, evaluated_gens, target_eval_percent)

            test_agent = MCTSAgent(gen, n_iters, set_temp_zero=True,
                                   tag=self._organizer.tag)
            ratings = np.array([self._evaluator.arena_ratings[ix] for ix in self._evaluated_ixs])
            init_rating_estimate = EvalUtils.estimate_rating_nearby_gens(gen, evaluated_gens,
                                                               ratings)
            self._evaluator.eval_agent(test_agent, n_games,
                                       error_threshold=error_threshold,
                                       init_rating_estimate=init_rating_estimate)
            new_ix = self._evaluator.agent_lookup[test_agent].index
            assert new_ix == len(self._evaluator.indexed_agents) - 1
            self._evaluated_ixs.append(new_ix)


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

        return None

    @staticmethod
    def get_next_gen_to_eval(latest_gen: int, evaluated_gens: List[int], target_eval_percent: float):
        if 0 not in evaluated_gens:
            return 0
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
