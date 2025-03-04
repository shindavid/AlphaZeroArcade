from alphazero.logic.agent_types import Agent, MCTSAgent, AgentRole
from alphazero.logic.arena import RatingData
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.ratings import win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger

from scipy.interpolate import interp1d
import numpy as np

from dataclasses import dataclass
from typing import List


logger = get_logger()


@dataclass
class EvalRatingData:
    evaluated_agents: List[Agent]
    ratings: np.ndarray


class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer):
        self._organizer = organizer
        self._benchmark = Benchmarker(organizer, db_filename=organizer.eval_db_filename)
        benchmark_rating_data = self._benchmark.read_ratings_from_db()
        self._benchmark_ratings = benchmark_rating_data.ratings
        self._benchmark_committee = benchmark_rating_data.committee

        self._arena = self._benchmark.clone_arena()
        self._db = RatingDB(self._organizer.eval_db_filename)
        self.load_from_db()

    def load_from_db(self):
        self._arena.load_agents_from_db(self._db, role=AgentRole.TEST)
        self._arena.load_matches_from_db(self._db, type=MatchType.EVALUATE)

    def eval_agent(self, test_agent: Agent, n_games, error_threshold=100):
        """
        Generic evaluation function for all types of agents.

        The opponent selection algorithm is adapted from KataGo:
        "Accelerating Self-Play Learning in Go" by David J. Wu (Section 5.1).

        The evaluation process follows these steps:

        1. Compute the test agent's probability of winning against each committee member
           based on the difference in their estimated Elo ratings. The initial estimated rating of
           the test agent is set to be the mean of the committee members' ratings if it has not
           played any matches yet.
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
        estimated_rating = np.mean(self._benchmark_ratings)
        test_iagent = self._arena._add_agent(test_agent, AgentRole.TEST, expand_matrix=True, db=self._db)

        n_games_played = self._arena.n_games_played(test_agent)
        if n_games_played > 0:
            n_games -= n_games_played
            estimated_rating = self._arena.ratings[test_iagent.index]

        committee_ixs = np.where(self._benchmark_committee)[0]
        opponent_ix_played = self._arena.get_past_opponents_ix(test_agent)
        while n_games > 0 and len(opponent_ix_played) < len(committee_ixs):

            p = [win_prob(estimated_rating, self._arena.ratings[ix]) for ix in committee_ixs]
            var = np.array([q * (1 - q) for q in p])
            mask = np.zeros(len(var), dtype=bool)
            committee_ix_played = np.where(np.isin(committee_ixs, opponent_ix_played))[0]
            mask[committee_ix_played] = True
            var[mask] = 0
            var = var / np.sum(var)

            sample_ixs = committee_ixs[np.random.choice(len(committee_ixs), p=var, size=n_games)]
            chosen_ixs, num_matches = np.unique(sample_ixs, return_counts=True)
            sorted_ixs = np.argsort(num_matches)[::-1]
            logger.info('evaluating %s against %d opponents. Estimated rating: %f', test_agent, len(chosen_ixs), estimated_rating)
            for i in range(len(chosen_ixs)):
                ix = chosen_ixs[sorted_ixs[i]]
                n = num_matches[sorted_ixs[i]]

                opponent = self._arena.indexed_agents[ix].agent
                match = Match(test_agent, opponent, n, MatchType.EVALUATE)
                self._arena.play_matches([match], self._organizer.game, db=self._db)
                n_games -= n
                opponent_ix_played = np.concatenate([opponent_ix_played, [ix]])
                self._arena.refresh_ratings()
                new_estimate = self._arena.ratings[test_iagent.index]
                if abs(new_estimate - estimated_rating) > error_threshold:
                    estimated_rating = new_estimate
                    break

        interpolated_ratings = self.interpolate_ratings()
        test_iagents = [ia for ia in self._arena.indexed_agents if ia.role == AgentRole.TEST]
        self._db.commit_ratings(test_iagents, interpolated_ratings)
        logger.info('Finished evaluating %s. Interpolated rating: %f', test_agent, interpolated_ratings[-1])

    def interpolate_ratings(self) -> np.ndarray:
        benchmark_ixs = self.benchmark_agent_ixs()
        test_ixs = self.test_agent_ixs()

        self._arena.refresh_ratings()
        xs = self._arena.ratings[benchmark_ixs][self._benchmark_committee]
        ys = self._benchmark_ratings[self._benchmark_committee]
        test_agents_elo = self._arena.ratings[test_ixs]
        sorted_ixs = np.argsort(xs)
        xs_sorted = xs[sorted_ixs]
        ys_sorted = ys[sorted_ixs]
        interp_func = interp1d(xs_sorted, ys_sorted, kind="linear", fill_value="extrapolate")
        interpolated_ratings = interp_func(test_agents_elo)

        return interpolated_ratings

    def test_agent_ixs(self) -> np.ndarray:
        test_ixs = [iagent.index for iagent in self._arena.indexed_agents if iagent.role == AgentRole.TEST]
        return np.array(test_ixs)

    def benchmark_agent_ixs(self) -> np.ndarray:
        benchmark_ixs = [iagent.index for iagent in self._arena.indexed_agents if iagent.role == AgentRole.BENCHMARK]
        return np.array(benchmark_ixs)

    def read_ratings_from_db(self) -> EvalRatingData:
        rating_data: RatingData = self._arena.load_ratings_from_db(self._db, AgentRole.TEST)
        ratings = rating_data.ratings
        evaluated_agents = [self._arena.agent_lookup_db_id[db_id].agent for db_id in rating_data.agent_ids]
        return EvalRatingData(evaluated_agents, ratings)


class MCTSEvaluator:
    def __init__(self, organizer: DirectoryOrganizer):
        self._organizer = organizer
        self._evaluator = Evaluator(organizer)
        rating_data = self._evaluator.read_ratings_from_db()
        self._evaluated_gens = [agent.gen for agent in rating_data.evaluated_agents]

    def run(self, n_iters: int=100, target_eval_percent: float=1.0, n_games: int=100, error_threshold=100):
        """
        Used for evaluating generations of MCTS agents of a run.
        """
        while True:
            gen = self.get_next_gen_to_eval(target_eval_percent)
            if gen is None:
                break

            test_agent = MCTSAgent(gen, n_iters, set_temp_zero=True,
                                   tag=self._organizer.tag)
            self._evaluator.eval_agent(test_agent, n_games, error_threshold)
            self._evaluated_gens.append(gen)

    def get_next_gen_to_eval(self, target_eval_percent):
        last_gen = self._organizer.get_latest_model_generation()
        evaluated_percent = len(self._evaluated_gens) / (last_gen + 1)
        if 0 not in self._evaluated_gens:
            return 0
        if last_gen not in self._evaluated_gens:
            return last_gen
        if evaluated_percent >= target_eval_percent:
            return None

        left_gen, right_gen = self.get_biggest_gen_gap()  # based on generation, not rating
        if left_gen + 1 < right_gen:
            gen = (left_gen + right_gen) // 2
            assert gen not in self._evaluated_gens
        return int(gen)

    def get_biggest_gen_gap(self):
        gens = self._evaluated_gens.copy()
        gens = np.sort(gens)
        gaps = np.diff(gens)
        max_gap_ix = np.argmax(gaps)
        left_gen = gens[max_gap_ix]
        right_gen = gens[max_gap_ix + 1]
        return left_gen, right_gen


