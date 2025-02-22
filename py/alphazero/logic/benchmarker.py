from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import Arena
from alphazero.logic.match_runner import Match
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger

import numpy as np

from dataclasses import dataclass
from typing import List, Optional


logger = get_logger()


@dataclass
class RatingsGap:
    left_gen: int
    right_gen: int
    elo_diff: float


class Benchmarker:
    """
    Manages a collection of Agents, their pairwise matches, and rating calculations.

    At any point, we will have R reference agents and M mcts agents.

    self.W_matrix: np.ndarray of shape (R+M, R+M)
    self.agents_lookup: Dict[Agent, int] of size R+M
    self.mcts_agents: List[MctsAgent] of size M, sorted by gen
    self.ratings: List[float] of size R+M
    """
    def __init__(self, organzier: DirectoryOrganizer, load_past_data: bool=False):
        self._organizer = organzier
        self.arena = Arena()
        self.ratings = None  # 1D np.ndarray
        self.ref_agents: List[Agent] = []

        if load_past_data:
            # maybe we always want to load past data?
            self.arena.load_matches_from_db(self._organizer.benchmark_db_filename)
            if not self.has_no_matches():
                self.compute_ratings()

    def run(self, n_iters: int=100, elo_threshold: int=100, n_games: int=100):
        while True:
            matches = self.get_next_matches(n_iters, elo_threshold, n_games)
            if matches is None:
                break
            counts = self.arena.play_matches(matches)
            self.compute_ratings()
            for match, count in zip(matches, counts):
                self.arena.commit_match_to_db(self._organizer.benchmark_db_filename,
                                              match, count)

        self.arena.commit_ratings_to_db(self._organizer.benchmark_db_filename,
                                        self.arena.agents_lookup.keys(), self.ratings)

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
        r = self.ratings.copy()
        mcts_agent_gens, mcts_agent_ix = zip(*[(agent.gen, ix) for agent, ix in \
            self.arena.agents_lookup.items() if agent not in self.ref_agents])
        mcts_agent_gens = np.array(mcts_agent_gens)
        mcts_agent_ix = np.array(mcts_agent_ix)

        mcts_ratings = r[mcts_agent_ix]
        # primary key is mcts_agent_gens, secondary key is mcts_ratings
        sorted_ix = np.argsort(mcts_agent_gens)
        gaps = np.abs(np.diff(mcts_ratings[sorted_ix]))

        sorted_gap_ix = np.argsort(gaps)[::-1]
        for gap_ix in sorted_gap_ix:
            left_gen = int(mcts_agent_gens[sorted_ix[gap_ix]])
            right_gen = int(mcts_agent_gens[sorted_ix[gap_ix + 1]])
            if left_gen + 1 < right_gen:
                return RatingsGap(left_gen, right_gen, float(gaps[gap_ix]))

    def compute_ratings(self, eps=1e-6):
        self.ratings = self.arena.compute_ratings(eps=eps)

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
        mid_agent = self.build_agent(mid_gen, n_iters)
        left_agent = self.build_agent(gap.left_gen, n_iters)
        right_agent = self.build_agent(gap.right_gen, n_iters)
        return [Match(left_agent, mid_agent, n_games),
                Match(mid_agent, right_agent, n_games)]

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

    def has_no_matches(self):
        return len(self.arena.agents_lookup) < 2

    @property
    def agents(self):
        return list(self.arena.agents_lookup.keys())




