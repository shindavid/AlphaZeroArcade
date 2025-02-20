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

    def load_past_data(self):
        # rows = self.rating_db.fetchall()
        # for gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws in rows:
        #     agent1 = RatingDB.build_agent_from_row(gen1, gen_iters1)
        #     agent2 = RatingDB.build_agent_from_row(gen2, gen_iters2)
        #     ix1, _ = self._add_agent_node(agent1)
        #     ix2, _ = self._add_agent_node(agent2)

        #     counts = WinLossDrawCounts(gen1_wins, gen2_wins, draws)
        #     self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
        #     self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

        wld_dict = {}
        rows = self.rating_db.fetchall()
        for gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws in rows:
            agent1 = RatingDB.build_agent_from_row(gen1, gen_iters1)
            agent2 = RatingDB.build_agent_from_row(gen2, gen_iters2)
            ix1, _ = self._add_agent_node(agent1, expand=False)
            ix2, _ = self._add_agent_node(agent2, expand=False)

            counts = WinLossDrawCounts(gen1_wins, gen2_wins, draws)
            wld_dict[(ix1, ix2)] = counts

        self._init_W_matrix(len(self.agent_ix))
        for (ix1, ix2), counts in wld_dict.items():
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

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

    def play_matches(self, matches: List[Match], additional=False):
        iterator = tqdm(matches) if len(matches) > 1 else matches
        for match in iterator:
            ix1, _ = self._add_agent_node(match.agent1)
            ix2, _ = self._add_agent_node(match.agent2)

            if self.W_matrix[ix1, ix2] > 0 or self.W_matrix[ix2, ix1] > 0:
                if not additional:
                    n_games_played = self.W_matrix[ix1, ix2] + self.W_matrix[ix2, ix1]
                    match.n_games = match.n_games - n_games_played
                if match.n_games < 1:
                    continue

            result = MatchRunner.run_match_helper(match, self.binary)
            self.W_matrix[ix1, ix2] += result.win + 0.5 * result.draw
            self.W_matrix[ix2, ix1] += result.loss + 0.5 * result.draw
            self.rating_db.commit_counts(match.agent1, match.agent2, result)

    def compute_ratings(self):
        assert self.W_matrix.shape[0] > 0
        assert np.all(transitive_closure(self.W_matrix))
        ratings = compute_ratings(self.W_matrix).tolist()
        self.ratings = {agent: ratings[ix] for agent, ix in self.agent_ix.items()}
        return self.ratings

    def sub_committee(self, include_agents: List[Agent]=None, exclude_agents: List[Agent]=None, \
        organizer: DirectoryOrganizer=None, db_name: str=None, binary: str=None)\
        -> 'BenchmarkCommittee':
        """
        Create a new BenchmarkCommittee for a subset of agents.

        This method filters the current committee's agents based on the
        optional lists provided, then constructs a new committee with the filtered data.
        Any match results in W_matrix are copied over for agents that remain.

        Args:
            include_agents (List[Agent], optional):
                If provided, only these agents (and edges between them) are considered.
                Defaults to including all agents if not specified.
            exclude_agents (List[Agent], optional):
                If provided, these agents (and edges involving them) are excluded from
                the new committee.
            organizer (DirectoryOrganizer, optional):
                A different `DirectoryOrganizer` for the new sub-committee. If omitted,
                the current committee's organizer is reused.
            db_name (str, optional):
                If provided, name of a new database for the sub-committee. Otherwise,
                the current committee's db_name is used.
            binary (str, optional):
                An alternate path to the game executable for the sub-committee. If omitted,
                the current binary is reused.

        Returns:
            BenchmarkCommittee:
                A new `BenchmarkCommittee` instance containing only the filtered subset
                of agents.
        """

        new_organizer = organizer if organizer else self._organizer
        new_db_name = db_name if db_name else self.db_name
        new_binary = binary if binary else self.binary
        sub_committee = BenchmarkCommittee(new_organizer, new_db_name, new_binary, load_past_data=False)

        for agent in self.agent_ix:
            if include_agents and exclude_agents:
                assert not (agent in include_agents and agent in exclude_agents)
            if exclude_agents and agent in exclude_agents:
                continue
            if not include_agents or agent in include_agents:
                new_ix, is_new_node = sub_committee._add_agent_node(agent)
                assert is_new_node
                assert sub_committee.W_matrix.shape[0] == new_ix + 1

        for agent_i, i in sub_committee.agent_ix.items():
            for agent_j, j in sub_committee.agent_ix.items():
                old_i = self.agent_ix[agent_i]
                old_j = self.agent_ix[agent_j]
                sub_committee.W_matrix[i, j] = self.W_matrix[old_i, old_j]

        return sub_committee



