from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import Arena
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.match_runner import Match
from alphazero.logic.ratings import win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger

from tqdm import tqdm
from scipy.interpolate import interp1d
import numpy as np

from typing import Optional, List



logger = get_logger()


class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_organizer: DirectoryOrganizer):
        self._organizer = organizer
        self.benchmark = Benchmarker(benchmark_organizer, load_past_data=True)
        self.benchmark_agents = self.benchmark.agents

        self.arena = Arena()
        self.arena.load_matches_from_db(benchmark_organizer.benchmark_db_filename)
        test_agents_with_matches = self.arena.load_matches_from_db(organizer.eval_db_filename)
        self.gens_with_matches = [agent.gen for agent in test_agents_with_matches]

        self.evaluated_gens: List[int] = []
        self.ratings: np.ndarray = np.array([])
        agent_ratings = self.arena.load_ratings_from_db(organizer.eval_db_filename)
        if agent_ratings:
            self.evaluated_agents, self.ratings = zip(*agent_ratings.items())
            self.evaluated_gens = [agent.gen for agent in self.evaluated_agents]
            self.ratings = np.array(self.ratings)

    def run(self, n_iters: int=100, target_eval_percent: float=1.0, n_games: int=100):
        while True:
            gen = self.get_next_gen_to_eval(target_eval_percent)
            if gen is None:
                break

            test_agent = MCTSAgent(gen, n_iters, set_temp_zero=True,
                                   binary_filename=self._organizer.binary_filename,
                                   model_filename=self._organizer.get_model_filename(gen))
            self.eval_agent(test_agent, n_games)

    def get_next_gen_to_eval(self, target_eval_percent):
        last_gen = self._organizer.get_latest_model_generation()
        evaluated_percent = len(self.evaluated_gens) / (last_gen + 1)
        if 0 not in self.evaluated_gens:
            return 0
        if last_gen not in self.evaluated_gens:
            return last_gen
        if evaluated_percent >= target_eval_percent:
            return None

        left_gen, right_gen = self.get_biggest_gen_gap()  # based on generation, not rating
        if left_gen + 1 < right_gen:
            gen = (left_gen + right_gen) // 2
            assert gen not in self.evaluated_gens
        return int(gen)

    def get_biggest_gen_gap(self):
        gens = self.evaluated_gens.copy()
        gens = np.sort(gens)
        gaps = np.diff(gens)
        max_gap_ix = np.argmax(gaps)
        left_gen = gens[max_gap_ix]
        right_gen = gens[max_gap_ix + 1]
        return left_gen, right_gen

    def eval_agent(self, test_agent: Agent, n_games):
        opponent_ix_played: np.ndarray = np.array([], dtype=int) # list of indices
        # arena will only have matches between the test agent and benchmark agents. Test agents
        # of a different gen will not be included in this arena. arena includes every benchmark agent.
        arena = self.arena.create_subset(include_agents=[test_agent] + self.benchmark_agents)

        estimated_rating = None
        if isinstance(test_agent, MCTSAgent):
            if test_agent.gen in self.gens_with_matches:
                ratings = arena.compute_ratings(eps=1e-3)
                estimated_rating = ratings[arena.agents_lookup[test_agent]]
                opponent_ix_played = arena.opponent_ix_played_against(test_agent)
                n_games -= arena.n_games_played(test_agent)

            else:
                estimated_rating = self.estimate_rating(test_agent.gen)  # interpolates based on rating of left_gen and right_gen

        if estimated_rating:
            ratings = arena.compute_ratings(eps=1e-3)
        else:
            # play with a few random opponents picking the highest variance
            opponent_ix = self.benchmark.committee_ix[np.random.choice(len(self.benchmark.committee_ix))]
            opponent = self.benchmark_agents[opponent_ix]
            match = Match(test_agent, opponent, n_games//10)
            counts = arena.play_matches([match], additional=False)
            arena.commit_match_to_db(self._organizer.eval_db_filename, match, counts[0])
            ratings = arena.compute_ratings(eps=1e-3)
            estimated_rating = ratings[arena.agents_lookup[test_agent]]
            opponent_ix_played = np.concatenate([opponent_ix_played, [opponent_ix]])

        if n_games > 0:
            p = [win_prob(estimated_rating, ratings[ix]) for ix in self.benchmark.committee_ix]
            var = np.array([q * (1 - q) for q in p])
            mask = np.zeros(len(var), dtype=bool)
            committee_ix_played = np.where(np.isin(self.benchmark.committee_ix, opponent_ix_played))[0]
            mask[committee_ix_played] = True
            var[mask] = 0
            var = var / np.sum(var)

            sample_ix = np.random.choice(len(self.benchmark.committee_ix), p=var, size=n_games)
            indices = [self.benchmark.committee_ix[ix] for ix in sample_ix]
            chosen_ix, num_matches = np.unique(indices, return_counts=True)
            print(f"evaluating {test_agent}:")
            for ix, n in tqdm(zip(chosen_ix, num_matches), total=len(chosen_ix)):
                opponent = self.benchmark_agents[ix]
                match = Match(test_agent, opponent, n)
                counts = arena.play_matches([match], additional=True)
                arena.commit_match_to_db(self._organizer.eval_db_filename, match, counts[0])

        ratings = arena.compute_ratings(eps=1e-3)
        rating = ratings[arena.agents_lookup[test_agent]]
        rating_benchmarked = self.interpolate_ratings(rating, arena)
        self.evaluated_gens.append(test_agent.gen)
        arena.commit_ratings_to_db(self._organizer.eval_db_filename, [test_agent], [rating_benchmarked])
        self.ratings = np.concatenate([self.ratings, [rating]])

    def estimate_rating(self, gen: int, max_gen_gap: int = 500) -> Optional[float]:
        assert gen not in self.evaluated_gens
        # find the closest gen that is less than gen and the closest gen that is greater than gen.
        # Then interpolate the rating.
        left_gen = max([g for g in self.evaluated_gens if g < gen], default=None)
        right_gen = min([g for g in self.evaluated_gens if g > gen], default=None)

        if not left_gen or not right_gen:
            return None

        assert left_gen < gen < right_gen
        if right_gen - left_gen > max_gen_gap:
            return None
        left_rating = self.ratings[self.evaluated_gens.index(left_gen)]
        right_rating = self.ratings[self.evaluated_gens.index(right_gen)]
        rating = np.interp(gen, [left_gen, right_gen], [left_rating, right_rating])
        return float(rating)

    def interpolate_ratings(self, estimated_rating: float, arena: Arena) -> float:
        ratings = arena.compute_ratings(eps=1e-3)
        x = []
        y = []
        for agent in self.benchmark_agents:
            x.append(ratings[arena.agents_lookup[agent]])
            y.append(self.benchmark.ratings[self.benchmark.agents_lookup[agent]])
        sorted_ix = np.argsort(x)
        x = np.array(x)[sorted_ix]
        y = np.array(y)[sorted_ix]
        interp_func = interp1d(x, y, kind="linear", fill_value="extrapolate")
        return float(interp_func(estimated_rating))

