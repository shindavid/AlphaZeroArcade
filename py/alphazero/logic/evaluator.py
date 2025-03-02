from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import RatingData
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.match_runner import Match
from alphazero.logic.ratings import win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger

from scipy.interpolate import interp1d
import numpy as np

from typing import List


logger = get_logger()


class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_organizer: DirectoryOrganizer):
        self._organizer = organizer
        self._db = RatingDB(self._organizer.eval_db_filename)
        self._benchmark = Benchmarker(benchmark_organizer)
        self._arena = self._benchmark.clone_arena()

        self._arena.load_agents_from_db(self._db)
        self._arena.load_matches_from_db(self._db)
        rating_data: RatingData = self._arena.load_ratings_from_db(self._db)

        self._evaluated_ixs = np.array([self._arena.agent_lookup_db_id[agent_id] for agent_id in rating_data.agent_ids])
        self._ratings = rating_data.ratings

    def eval_agent(self, test_agent: Agent, n_games, error_threshold=100):
        """
        generic evaluation function for all types of agent.
        """
        self._arena.refresh_ratings()
        estimated_rating = np.mean(self._benchmark.ratings)
        iagent = self._arena._add_agent(test_agent, expand_matrix=True, db=self._db)
        n_games -= self._arena.n_games_played(test_agent)
        if iagent.index in self._evaluated_ixs:
            estimated_rating = self._arena.ratings[iagent.index]

        opponent_ix_played = self._arena.get_past_opponents_ix(test_agent)
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
            p = [win_prob(estimated_rating, self._arena.ratings[ix]) for ix in self.committee_ixs]
            var = np.array([q * (1 - q) for q in p])
            mask = np.zeros(len(var), dtype=bool)
            committee_ix_played = np.where(np.isin(self.committee_ixs, opponent_ix_played))[0]
            mask[committee_ix_played] = True
            var[mask] = 0
            var = var / np.sum(var)

            sample_ixs = self.committee_ixs[np.random.choice(len(self.committee_ixs), p=var, size=n_games)]
            chosen_ixs, num_matches = np.unique(sample_ixs, return_counts=True)
            sorted_ixs = np.argsort(num_matches)[::-1]
            logger.info('evaluating %s against %d opponents. Estimated rating: %f', test_agent, len(chosen_ixs), estimated_rating)
            for i in range(len(chosen_ixs)):
                ix = chosen_ixs[sorted_ixs[i]]
                n = num_matches[sorted_ixs[i]]

                opponent = self.indexed_agents[ix].agent
                match = Match(test_agent, opponent, n)
                self._arena.play_matches([match], self._organizer.game, additional=True, db=self._db)
                n_games -= n
                opponent_ix_played = np.concatenate([opponent_ix_played, [ix]])
                self._arena.refresh_ratings()
                new_estimate = self._arena.ratings[iagent.index]
                if abs(new_estimate - estimated_rating) > error_threshold:
                    estimated_rating = new_estimate
                    break

        self._arena.refresh_ratings()
        eval_rating = self._arena.ratings[iagent.index]
        interpolated_rating = self.interpolate_ratings(eval_rating)

        logger.info('%s rating: %f, raw: %f', test_agent, interpolated_rating, eval_rating)
        self._evaluated_ixs = np.concatenate([self._evaluated_ixs, [iagent.index]])
        self._ratings = np.concatenate([self._ratings, [interpolated_rating]])
        self._db.commit_ratings([iagent], [interpolated_rating])

    def interpolate_ratings(self, estimated_rating: float) -> float:
        x = self._benchmark.ratings
        y = self._arena.ratings[:len(self._benchmark.ratings)]
        sorted_ixs = np.argsort(x)
        x_sorted = x[sorted_ixs]
        y_sorted = y[sorted_ixs]
        interp_func = interp1d(x_sorted, y_sorted, kind="linear", fill_value="extrapolate")
        return float(interp_func(estimated_rating))

    @property
    def indexed_agents(self):
        return self._arena.indexed_agents

    @property
    def committee(self):
        return self._benchmark.committee

    @property
    def evaluated_ixs(self):
        return self._evaluated_ixs

    @property
    def ratings(self):
        return self._ratings


class MCTSEvaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_organizer: DirectoryOrganizer):
        self._organizer = organizer
        self._evaluator = Evaluator(organizer, benchmark_organizer)
        self._evaluated_gens: List[int] = [self.indexed_agents[ix].agent.gen for ix in self.evaluated_ixs]

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

    @property
    def indexed_agents(self):
        return self._evaluator.indexed_agents

    @property
    def evaluated_ixs(self):
        return self._evaluator.evaluated_ixs

