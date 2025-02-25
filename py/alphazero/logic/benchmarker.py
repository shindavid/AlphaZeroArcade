from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import Arena
from alphazero.logic.match_runner import Match
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.rating_db import RatingDB
from util.logging_util import get_logger

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import numpy as np
from typing import  Optional, List


logger = get_logger()


@dataclass
class RatingsGap:
    left_gen: int
    right_gen: int
    elo_diff: float


class Benchmarker:
    """
    Manages a collection of Agents, their pairwise matches, and rating calculations.
    """
    def __init__(self, organizer: DirectoryOrganizer):
        self._organizer = organizer
        self.arena = Arena()
        self.ratings: np.ndarray = np.array([])
        self.committee_ix: np.ndarray = np.array([])
        self._database = RatingDB(self._organizer.benchmark_db_filename)
        self.load_from_db()

    def load_from_db(self):
        self.ratings, self.committee_ix = self.arena.load_ratings_from_db(self._database)
        self.arena.load_matches_from_db(self._database)
        if not self.has_no_matches():
            self.compute_ratings()

    def run(self, n_iters: int=100, elo_threshold: int=100, n_games: int=100, elo_gap: int=200):
        while True:
            matches = self.get_next_matches(n_iters, elo_threshold, n_games)
            if matches is None:
                break
            self.arena.play_matches(matches, additional=False, db=self._database)
            self.compute_ratings()

        is_committee = self.select_committee(elo_gap)
        self.compute_ratings()
        self.arena.commit_ratings_to_db(self.agents.values(), self.ratings, is_committee)

    def get_next_matches(self, n_iters, elo_threshold, n_games):
        if self.has_no_matches():
            gen0_agent = self.build_agent(0, n_iters)
            last_gen = self._organizer.get_latest_model_generation()
            last_gen_agent = self.build_agent(last_gen, n_iters)
            return [Match(gen0_agent, last_gen_agent, n_games)]

        gap = self.get_biggest_mcts_ratings_gap()
        if gap is None or gap.elo_diff < elo_threshold:
            return None
        mid_gen = (gap.left_gen + gap.right_gen) // 2

        logger.info(f'Adding mid point: gen-{mid_gen}')
        mid_agent = self.build_agent(mid_gen, n_iters)
        matches = [Match(mid_agent, agent, n_games) for agent in self.agents.values()]
        return matches

    def get_biggest_mcts_ratings_gap(self) -> Optional[RatingsGap]:
        """
        At any point, the sorted list of mcts-ratings looks like:

        gen_1, rating_1
        gen_2, rating_2
        ...
        gen_n, rating_n

        with rating_1 <= rating_2 <= ... <= rating_n

        Among all i in the range [1...n] with the property that g_i + 1 < G_i, identifies the one
        for which rating_{i_1} - rating_i is maximal, and returns the corresponding gap.

        Here,

        g_i = min(gen_i, gen_{i+1})
        G_i = max(gen_i, gen_{i+1})

        If no such i exists, returns None.

        NOTE: if we want to do partial-gens (i.e., use -i/--num-mcts-iters), then we can change this
        appropriately.
        """
        elos = self.ratings.copy()
        gens, agent_ix = zip(*[(agent.gen, ix) for ix, agent in self.agents.items()])
        gens = np.array(gens)
        agent_ix = np.array(agent_ix)

        sorted_ix = np.argsort(gens)
        gaps = np.abs(np.diff(elos[sorted_ix]))
        sorted_gap_ix = np.argsort(gaps)[::-1]
        for gap_ix in sorted_gap_ix:
            left_gen = int(gens[sorted_ix[gap_ix]])
            right_gen = int(gens[sorted_ix[gap_ix + 1]])
            if left_gen + 1 < right_gen:
                return RatingsGap(left_gen, right_gen, float(gaps[gap_ix]))

    def compute_ratings(self, eps=1e-3):
        self.ratings = self.arena.compute_ratings(eps=eps)

    def build_agent(self, gen: int, n_iters):
        if gen == 0:
            return MCTSAgent(gen=gen,
                             n_iters=n_iters,
                             binary_filename=self._organizer.binary_filename)
        else:
            return MCTSAgent(gen=gen,
                             n_iters=n_iters,
                             set_temp_zero=True,
                             binary_filename=self._organizer.binary_filename,
                             model_filename=self._organizer.get_model_filename(gen))

    def select_committee(self, elo_gap) -> List[bool]:
        elos = self.ratings.copy()
        max_elo = np.max(elos)
        min_elo = np.min(elos)
        committee_size = int((max_elo - min_elo) / elo_gap)
        target_elos = np.linspace(min_elo, max_elo, committee_size)

        committee_ix = []
        j = 0
        n = len(elos)
        for e in target_elos:
            while j < n - 1 and (abs(elos[j] - e) > abs(elos[j + 1] - e)):
                j += 1
            committee_ix.append(j)
        committee_ix = np.array(committee_ix)
        self.committee_ix = committee_ix

        mask = np.zeros(len(elos), dtype=bool)
        mask[committee_ix] = True
        return mask.astype(int).tolist()

    def has_no_matches(self):
        return np.sum(self.arena.W_matrix) == 0

    @property
    def agents(self):
        return self.arena.agents

