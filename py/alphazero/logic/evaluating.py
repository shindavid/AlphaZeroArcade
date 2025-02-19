from alphazero.logic.agent_types import Agent
from alphazero.logic.benchmarking import  BenchmarkCommittee
from alphazero.logic.match_runner import Match
from alphazero.logic.ratings import BETA_SCALE_FACTOR
from alphazero.logic.rating_db import RatingDB
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

from tqdm import tqdm
import numpy as np

from typing import Dict
import random

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

    def evaluate(self, test_agent: Agent, n_games: int=10, n_steps: int=10) -> float:
        representatives = []
        init_benchmark_agent = random.choice(list(self.benchmark_committee.G.nodes))
        init_match = Match(test_agent, init_benchmark_agent, n_games)
        self.eval.play_matches([init_match], additional=True)
        representatives.append(init_benchmark_agent)
        for _ in tqdm(range(n_steps)):
            eval_sub_committee = self.eval.sub_committee(include_agents=[test_agent] + list(self.benchmark_committee.G.nodes))
            eval_sub_committee.compute_ratings()
            eval_ratings = eval_sub_committee.ratings
            p = {agent: 1/(1 + np.exp((rating - eval_ratings[test_agent])/BETA_SCALE_FACTOR)) \
                for agent, rating in eval_ratings.items() if agent != test_agent}
            weights = {agent: p * (1 - p) for agent, p in p.items()}
            next_agent = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            next_match = Match(test_agent, next_agent, n_games)
            self.eval.play_matches([next_match], additional=True)
            representatives.append(next_agent)

        test_group_elo_ratings = eval_sub_committee.ratings
        test_rating = self.interpolate_ratings(test_agent, test_group_elo_ratings)
        self.eval.rating_db.commit_rating(test_agent, test_rating, representatives, self._organizer.tag)
        return test_rating

