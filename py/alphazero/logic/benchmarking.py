from alphazero.logic.agent_types import Agent
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.graph_util import transitive_closure
from util.logging_util import get_logger

import numpy as np
from tqdm import tqdm

from typing import List, Dict


logger = get_logger()


class Benchmarker:
    """
    Manages a collection of Agents, their pairwise matches, and rating calculations.

    This class tracks agents in a networkx.Graph (self.G) where each node is an
    Agent and edges represent the matches played between those agents. It also keeps
    a dense result matrix (self.W_matrix) of shape (N, N), where N is the number
    of unique agents in the committee. For agents i and j, W_matrix[i, j]
    records the total "partial wins" of agent i over agent j (i.e., wins plus half
    of the draws). The class can load past match data from a database and incrementally
    update it by scheduling and playing new matches.

    At any point, we will have R reference agents and M mcts agents.

    self.W_matrix: np.ndarray of shape (R+M, R+M)
    self.agents_lookup: Dict[Agent, int] of size R+M
    self.mcts_agents: List[MctsAgent] of size M, sorted by gen
    self.ratings: List[float] of size R+M
    """

    def __init__(self, organzier: DirectoryOrganizer, db_name, binary: str=None,\
        load_past_data: bool=False):

        self._organizer = organzier
        self.db_name = db_name

        self.arena = Arena()

        self.W_matrix = np.zeros((0, 0), dtype=float)
        self.agent_ix: Dict[Agent, int] = {}

        self.rating_db = RatingDB(self._organizer.benchmark_db_filename)
        self.binary = binary if binary else 'target/Release/bin/' + self._organizer.game
        self.ratings = None  # 1D np.ndarray

        if load_past_data:
            self.load_past_data()

    def run(self):
        while True:
            match = self.get_next_match()
            if match is None:
                break
            self.run_match(match)

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
        pass

    def get_next_match(self):
        if self.no_matches_yet():
            last_gen = self.organizer.get_latest_model_generation()
            return self.create_match(0, last_gen)

        gap = self.get_biggest_mcts_ratings_gap()
        if gap is None or gap.elo_differential < self.elo_threshold:
            return None

        return self.create_match(gap.left_gen, gap.right_gen)

    def compute_ratings(self):
        assert self.W_matrix.shape[0] > 0
        assert np.all(transitive_closure(self.W_matrix))
        ratings = compute_ratings(self.W_matrix).tolist()
        self.ratings = {agent: ratings[ix] for agent, ix in self.agent_ix.items()}
        return self.ratings



