from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import Arena
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.match_runner import Match
from alphazero.logic.ratings import win_prob
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

from tqdm import tqdm
import numpy as np

from collections import Counter
from typing import Dict, List
import random


class Evaluator:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_organizer: DirectoryOrganizer):
        self._organizer = organizer
        self.benchmark = Benchmarker(benchmark_organizer, load_past_data=True)
        self.benchmark_agents = self.benchmark.agents

        self.arena = Arena()
        self.arena.load_matches_from_db(benchmark_organizer.benchmark_db_filename)
        test_agents_with_matches = self.arena.load_matches_from_db(organizer.eval_db_filename)
        self.gens_with_matches = [agent.gen for agent in test_agents_with_matches]

        self.evaluated_gens, self.ratings = zip(*self.arena.load_ratings_from_db(organizer.eval_db_filename))
        self.evaluated_gens = list(self.evaluated_gens)
        self.ratings = np.array(self.ratings)

    def run(self, n_iters: int=100, target_eval_percent: float=1.0, n_games: int=100, n_steps: int=10):
        while True:
            gen = self.get_next_gen_to_eval(target_eval_percent)
            if gen is None:
                break

            eval_agent = MCTSAgent(gen, n_iters, set_temp_zero=True,
                                   binary_filename=self._organizer.binary_filename,
                                   model_filename=self._organizer.get_model_filename(gen))
            rating = self.eval_agent(eval_agent, n_games, n_steps)

            self.arena.commit_ratings_to_db(self._organizer.eval_db_filename, [gen], [rating])

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
        return gen

    def get_biggest_gen_gap(self):
        gens = self.evaluated_gens.copy()
        gens = np.sort(gens)
        gaps = np.diff(gens)
        max_gap_ix = np.argmax(gaps)
        left_gen = gens[max_gap_ix]
        right_gen = gens[max_gap_ix + 1]
        return left_gen, right_gen

    def eval_agent(self, test_agent: Agent, n_games, n_steps):
        opponent_ix_played: List[int] = [] # list of indices
        # arena will only have matches between the test agent and benchmark agents. Test agents
        # of a different gen will not be included in this arena. arena includes every benchmark agent.
        arena = self.arena.create_subset(include_agents=[test_agent] + self.benchmark_agents)

        if isinstance(test_agent, MCTSAgent):
            if test_agent.gen in self.gens_with_matches:
                ratings = arena.compute_ratings()
                estimated_rating = ratings[arena.agents_lookup[test_agent]]
                opponent_ix_played = arena.opponent_ix_played_against(test_agent)
                n_steps -= len(opponent_ix_played)

            elif len(self.evaluated_gens) >= 2:
                estimated_rating = self.estimate_rating(test_agent.gen)  # interpolates based on rating of left_gen and right_gen
        else:
            # play a random match against a benchmark agent
            opponent_ix = np.random.choice(len(self.benchmark_agents))
            opponent = self.benchmark_agents[opponent]
            arena.play_matches([Match(test_agent, opponent, n_games)], additional=False)
            ratings = arena.compute_ratings()
            estimated_rating = ratings[arena.agents_lookup[test_agent]]
            opponent_ix_played.append(opponent_ix)


        for _ in tqdm(range(n_steps)):
            p = [win_prob(estimated_rating, ratings[arena.agents_lookup[agent]]) \
                for agent in self.benchmark_agents]
            var = [q * (1 - q) for q in p]

            opponent_ix = np.random.choice(len(self.benchmark_agents), p=var)
            opponent = self.benchmark_agents[opponent_ix]
            match = Match(test_agent, opponent, n_games)
            counts = arena.play_matches([match], additional=False)
            self.arena.commit_match_to_db(self._organizer.eval_db_filename, match, counts[0])

            ratings = arena.compute_ratings()
            estimated_rating = ratings[arena.agents_lookup[test_agent]]
            opponent_ix_played.append(opponent_ix)

        rating = self.interploate_ratings(rating, arena)
        arena.commit_ratings_to_db(self._organizer.eval_db_filename, [test_agent.gen], [rating])

    def estimate_rating(self, gen: int) -> float:
        assert gen not in self.evaluated_gens
        # find the closest gen that is less than gen and the closest gen that is greater than gen.
        # Then interpolate the rating.
        left_gen = max([g for g in self.evaluated_gens if g < gen])
        right_gen = min([g for g in self.evaluated_gens if g > gen])
        left_rating = self.ratings[self.evaluated_gens.index(left_gen)]
        right_rating = self.ratings[self.evaluated_gens.index(right_gen)]
        rating = np.interp(gen, [left_gen, right_gen], [left_rating, right_rating])
        return float(rating)

    def intepolate_ratings(self, estimated_rating: float, arena: Arena) -> float:
        ratings = arena.compute_ratings()
        x = []
        y = []
        for agent in self.benchmark_agents:
            x.append(ratings[arena.agents_lookup[agent]])
            y.append(self.benchmark.ratings[self.benchmark.agents_lookup[agent]])
        return float(np.interp(estimated_rating, x, y))

class Evaluation:
    """
    Evaluates test agents against a benchmark committee.

    This class uses a reference BenchmarkCommittee that has already computed
    ratings for a set of benchmark Agents. It then creates a sub-committee for
    playing additional matches involving a new test , saves the match results to its own
    database, and finally returns an interpolated rating for the test Agent
    on the original benchmark's rating scale.
    """
    def __init__(self, organizer: DirectoryOrganizer, benchmark_committee: BenchmarkCommittee,\
        db_name, binary: str=None):

        self._organizer = organizer
        self.db_name = db_name
        self.binary = binary if binary else 'target/Release/bin/' + self._organizer.game
        self.benchmark_committee = benchmark_committee
        assert self.benchmark_committee.ratings is not None
        self.rating_db = RatingDB(self._organizer.databases_dir, self.db_name)
        self.eval = self.benchmark_committee.sub_committee(organizer=self._organizer,\
            db_name=self.db_name, binary=self.binary)
        self.benchmark_ratings = dict(sorted(self.benchmark_committee.ratings.items(), key=lambda x: x[1]))

    def interpolate_ratings(self, test_agent: Agent, test_group_elo_ratings: Dict[Agent, float])\
        -> float:
        """
        Interpolates the test agent's rating from a local sub-committee scale to the original
        benchmark committee's rating scale.

        The approach maps the test agent's rating in test_group_elo_ratings onto the known
        benchmark rating scale by linear interpolation.
        """

        interp_table = {v: self.benchmark_committee.ratings[k] for k, v in test_group_elo_ratings.items() if k != test_agent}
        interp_table = sorted(interp_table.items(), key=lambda x: x[1])
        interp_table = list(zip(*interp_table))
        x_values = interp_table[0]
        y_values = interp_table[1]
        x = test_group_elo_ratings[test_agent]
        test_rating = np.interp(x, x_values, y_values)
        return test_rating

    def evaluate(self, test_agent: Agent, n_games: int=100, n_steps: int=10) -> float:
        representatives = []
        init_benchmark_agent = random.choice(list(self.benchmark_committee.agent_ix.keys()))
        init_match = Match(test_agent, init_benchmark_agent, n_games)
        self.eval.play_matches([init_match], additional=True)
        representatives.append(init_benchmark_agent)
        # for _ in tqdm(range(n_steps)):
        if True:
            eval_sub_committee = self.eval.sub_committee(include_agents=[test_agent] + list(self.benchmark_committee.agent_ix.keys()))
            eval_sub_committee.compute_ratings()
            eval_ratings = eval_sub_committee.ratings
            p = {agent: 1/(1 + np.exp((rating - eval_ratings[test_agent])/BETA_SCALE_FACTOR)) \
                for agent, rating in eval_ratings.items() if agent != test_agent}

            agents = list(p.keys())
            weights = [q * (1-q) for q in p.values()]

            A = len(agents)
            # choices will be a length-100 np.ndarray of ints in the range [0, A)
            choices = np.random.choice(A, 100, p=weights)

            # TODO: Can use np.unique here instead
            counts = Counter()
            for c in choices:
                counts[c] += 1

            for agent_ix, n in counts.items():
                agent = agents[agent_ix]
                match = Match(test_agent, agent, n)
                TODO

            # weights = {agent: p * (1 - p) for agent, p in p.items()}
            # next_agent = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            # next_match = Match(test_agent, next_agent, n_games)
            # self.eval.play_matches([next_match], additional=True)
            # representatives.append(next_agent)



        test_group_elo_ratings = eval_sub_committee.ratings
        test_rating = self.interpolate_ratings(test_agent, test_group_elo_ratings)
        self.eval.rating_db.commit_rating(test_agent, test_rating, representatives, self._organizer.tag)
        return test_rating

