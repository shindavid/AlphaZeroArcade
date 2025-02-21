from alphazero.logic.agent_types import Agent
from alphazero.logic.benchmarker import  Benchmarker
from alphazero.logic.match_runner import Match
from alphazero.logic.ratings import BETA_SCALE_FACTOR
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

from tqdm import tqdm
import numpy as np

from collections import Counter
from typing import Dict
import random


class Evaluator:
    def __init__(self):
        self.arena = Arena()
        self.arena.load_past_matches(benchmark_organizer.benchmark_db_filename)
        self.arena.load_past_matches(organizer.eval_db_filename)

    def run(self):
        while True:
            gen = self.get_next_gen_to_eval()
            if gen is None:
                break
            self.eval_gen(gen)

    def get_next_gen_to_eval(self):
        if not self.evaluated(0):
            return 0
        if not self.evaluated(self.last_gen):
            return self.last_gen

        gap = self.get_biggest_gen_gap()  # based on generation, not rating
        gen = (gap.left_gen + gap.right_gen) // 2
        return gen

    def eval_gen(self, gen):
        # TODO: if resuming after loading past matches, we might already have some matches for
        # this gen, so we can skip this estimated-rating stuff

        estimated_rating = self.estimate_rating(gen)  # interpolates based on rating of left_gen and right_gen
        opponent = self.get_committee_member_of_similar_skill(estimated_rating)
        self.run_match(gen, opponent)  # updates ratings

        # Now, play remaining committee members using variance-calc p*(1-p)
        #
        # Question: do we recompute ratings after each match? Doing so makes the order of the loop
        # matter.
        #
        # I think we should match US Law here and prohibit "Double Jeopardy". In other words, if
        # we ask whether to face committee member X, using the current ratings to get a probability
        # p, and then flip our coin to say "do not run match", we cannot in the future decide to
        # flip that coin again. Rejecting X is permanent.


    def get_next_match(self):
        if not self.gen0_test_complete():
            pass

        if self.no_matches_yet():
            last_gen = self.organizer.get_latest_model_generation()
            return self.create_match(0, last_gen)

        gap = self.get_biggest_mcts_ratings_gap()
        if gap is None or gap.elo_differential < self.elo_threshold:
            return None

        return self.create_match(gap.left_gen, gap.right_gen)


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

