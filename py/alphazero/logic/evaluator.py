from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import Arena
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.match_runner import Match
from alphazero.logic.ratings import win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger

from scipy.interpolate import interp1d
import numpy as np

from typing import Optional, List


logger = get_logger()


class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_organizer: DirectoryOrganizer):
        self._organizer = organizer
        self.benchmark = Benchmarker(benchmark_organizer)
        self._db = RatingDB(self._organizer.eval_db_filename)

        self.arena = self.benchmark.arena.clone()
        self.arena.load_agents_from_db(self._db)
        self.arena.load_matches_from_db(self._db)
        self.evaluated_ixs, self.ratings, _ = self.arena.load_ratings_from_db(self._db)
        self.evaluated_versions: List[int] = [self.agents[ix].version for ix in self.evaluated_ixs]

    def run(self, n_iters: int=100, target_eval_percent: float=1.0, n_games: int=100, error_threshold=100):
        """
        Used for evaluating generations of MCTS agents of a run.
        """
        while True:
            gen = self.get_next_gen_to_eval(target_eval_percent)
            if gen is None:
                break

            test_agent = MCTSAgent(gen, n_iters, set_temp_zero=True,
                                   binary_filename=self._organizer.binary_filename,
                                   model_filename=self._organizer.get_model_filename(gen))
            self.eval_agent(test_agent, n_games, error_threshold)

    def get_next_gen_to_eval(self, target_eval_percent):
        last_gen = self._organizer.get_latest_model_generation()
        evaluated_percent = len(self.evaluated_versions) / (last_gen + 1)
        if 0 not in self.evaluated_versions:
            return 0
        if last_gen not in self.evaluated_versions:
            return last_gen
        if evaluated_percent >= target_eval_percent:
            return None

        left_gen, right_gen = self.get_biggest_gen_gap()  # based on generation, not rating
        if left_gen + 1 < right_gen:
            gen = (left_gen + right_gen) // 2
            assert gen not in self.evaluated_versions
        return int(gen)

    def get_biggest_gen_gap(self):
        gens = self.evaluated_versions.copy()
        gens = np.sort(gens)
        gaps = np.diff(gens)
        max_gap_ix = np.argmax(gaps)
        left_gen = gens[max_gap_ix]
        right_gen = gens[max_gap_ix + 1]
        return left_gen, right_gen

    def eval_agent(self, test_agent: Agent, n_games, error_threshold=100):
        """
        generic evaluation function for all types of agent.
        max_version_gap is used to determine if we can interpolate the rating given the rated
        agents in arena.
        """
        self.arena.compute_ratings()
        estimated_rating = np.mean(self.benchmark.ratings)
        ix, is_new = self.arena._add_agent(test_agent, expand_matrix=True, db=self._db)
        if not is_new:
            n_games -= self.arena.n_games_played(test_agent)
            estimated_rating = self.arena.ratings[ix]

        opponent_ix_played = []
        while n_games > 0 and len(opponent_ix_played) < len(self.committee_ixs):
            """
            The algorithm for selecting opponents is as follows:
            1. Select opponents from the committee proportionally to the variance of the win
            probability of the test agent against the opponent.
            2. If the test agent has played against an opponent, remove that opponent from the
            sampling pool.
            3. If the test agent's new estimated rating after including the newly played match is
            not within error_threshold of the original estimate, we conclude the estimate rating used
            was too off to give us a good basis for sampling opponents. We go back to step 1.
            4. If the test agent has played against all opponents or enough matches have been played,
            we stop and compute the final rating.
            5. The final rating is interpolated from the benchmark committee ratings before any games
            were played against the test agent.
            """
            p = [win_prob(estimated_rating, self.arena.ratings[ix]) for ix in self.committee_ixs]
            var = np.array([q * (1 - q) for q in p])
            mask = np.zeros(len(var), dtype=bool)
            committee_ix_played = np.where(np.isin(self.committee_ixs, opponent_ix_played))[0]
            mask[committee_ix_played] = True
            var[mask] = 0
            var = var / np.sum(var)

            sample_ixs = self.committee_ixs[np.random.choice(len(self.committee_ixs), p=var, size=n_games)]
            chosen_ixs, num_matches = np.unique(sample_ixs, return_counts=True)
            sorted_ixs = np.argsort(num_matches)[::-1]
            logger.info(f'evaluating {test_agent} against {len(chosen_ixs)} opponents. Estimated rating: {estimated_rating}')
            for i in range(len(chosen_ixs)):
                ix = chosen_ixs[sorted_ixs[i]]
                n = num_matches[sorted_ixs[i]]

                opponent = self.agents[ix]
                match = Match(test_agent, opponent, n)
                self.arena.play_matches([match], additional=True, db=self._db)
                n_games -= n
                opponent_ix_played.append(ix)
                self.arena.compute_ratings()
                new_estimate = self.arena.ratings[test_agent.ix]
                if abs(new_estimate - estimated_rating) > error_threshold:
                    estimated_rating = new_estimate
                    break

        self.arena.compute_ratings()
        eval_rating = self.arena.ratings[test_agent.ix]
        interpolated_rating = self.interpolate_ratings(eval_rating)

        logger.info(f'{test_agent} rating: {interpolated_rating}, before_interpolation: {eval_rating}')
        self.evaluated_ixs = np.concatenate([self.evaluated_ixs, [test_agent.ix]])
        self.ratings = np.concatenate([self.ratings, [interpolated_rating]])
        self.evaluated_versions.append(test_agent.version)
        self.arena.commit_ratings_to_db(self._db, [test_agent], [interpolated_rating])

    def interpolate_ratings(self, estimated_rating: float) -> float:
        x = self.benchmark.ratings
        y = self.arena.ratings[:len(self.benchmark.ratings)]
        interp_func = interp1d(x, y, kind="linear", fill_value="extrapolate")
        return float(interp_func(estimated_rating))

    @property
    def agents(self):
        return self.arena.agents

    @property
    def committee_ixs(self):
        return self.benchmark.committee_ixs