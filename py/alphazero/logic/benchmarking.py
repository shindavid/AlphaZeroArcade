from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import get_logger

import networkx as nx
import numpy as np
from tqdm import tqdm

import os
from typing import List, Tuple

logger = get_logger()

class BenchmarkCommittee:
    """
    Manages a collection of Agents, their pairwise matches, and rating calculations.

    This class tracks agents in a networkx.Graph (self.G) where each node is an
    Agent and edges represent the matches played between those agents. It also keeps
    a dense result matrix (self.W_matrix) of shape (N, N), where N is the number
    of unique agents in the committee. For agents i and j, W_matrix[i, j]
    records the total "partial wins" of agent i over agent j (i.e., wins plus half
    of the draws). The class can load past match data from a database and incrementally
    update it by scheduling and playing new matches.

    """

    def __init__(self, organzier: DirectoryOrganizer, db_name, binary: str=None,\
        load_past_data: bool=False):

        self._organizer = organzier
        self.db_name = db_name

        self.W_matrix = np.zeros((0, 0), dtype=float)
        self.G = nx.Graph()

        self.rating_db = RatingDB(self._organizer.databases_dir, self.db_name)
        self.binary = binary if binary else 'target/Release/bin/' + self._organizer.game
        self.ratings = None

        if load_past_data:
            self.load_past_data()

    def load_past_data(self):
        rows = self.rating_db.fetchall()
        for gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws in rows:
            agent1 = RatingDB.build_agent_from_row(gen1, gen_iters1, organizer=self._organizer)
            agent2 = RatingDB.build_agent_from_row(gen2, gen_iters2, organizer=self._organizer)
            num_games = gen1_wins + gen2_wins + draws
            ix1, _ = self._add_agent_node(agent1)
            ix2, _ = self._add_agent_node(agent2)

            if self.G.has_edge(agent1, agent2):
                self.G.edges[(agent1, agent2)]['n_games'] += num_games
            else:
                self.G.add_edge(agent1, agent2, n_games=num_games)

            counts = WinLossDrawCounts(gen1_wins, gen2_wins, draws)
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def play_matches(self, matches: List[Match], additional=False):
        iterator = tqdm(matches) if len(matches) > 1 else matches
        for match in iterator:
            ix1, _ = self._add_agent_node(match.agent1)
            ix2, _ = self._add_agent_node(match.agent2)

            if self.G.has_edge(match.agent1, match.agent2):
                if not additional:
                    n_games_played = self.G.edges[(match.agent1, match.agent2)]['n_games']
                    match.n_games = match.n_games - n_games_played
                if match.n_games < 1:
                    continue
            else:
                self.G.add_edge(match.agent1, match.agent2, n_games=0)

            result = MatchRunner.run_match_helper(match, self.binary)
            self.W_matrix[ix1, ix2] += result.win + 0.5 * result.draw
            self.W_matrix[ix2, ix1] += result.loss + 0.5 * result.draw
            self.G[match.agent1][match.agent2]['n_games'] += match.n_games
            self.rating_db.commit_counts(match.agent1, match.agent2, result)

    def compute_ratings(self):
        assert not nx.is_empty(self.G)
        assert nx.is_connected(self.G)
        ratings = compute_ratings(self.W_matrix).tolist()
        self.ratings = {agent: ratings[ix] for agent, ix in self.G.nodes(data='ix')}
        return self.ratings

    def sub_committee(self, include_agents: List[Agent]=None, exclude_agents: List[Agent]=None, \
        exclude_edges: List[Tuple[Agent, Agent]]=None, organizer: DirectoryOrganizer=None,\
            db_name: str=None, binary: str=None)\
            -> 'BenchmarkCommittee':
        """
        Create a new BenchmarkCommittee for a subset of agents and edges.

        This method filters the current committee's agents and edges based on the
        optional lists provided, then constructs a new committee with the filtered data.
        Any match results in `W_matrix` are copied over for agents (and edges) that remain.

        Args:
            include_agents (List[Agent], optional):
                If provided, only these agents (and edges between them) are considered.
                Defaults to including all agents if not specified.
            exclude_agents (List[Agent], optional):
                If provided, these agents (and edges involving them) are excluded from
                the new committee.
            exclude_edges (List[Tuple[Agent, Agent]], optional):
                A list of edges to exclude (in addition to those filtered out by agent
                exclusion). Each tuple is (agentA, agentB). Defaults to None.
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
                of agents and edges. Its rating matrix and graph contain data
                corresponding to the included set of agents.
        """

        new_organizer = organizer if organizer else self._organizer
        new_db_name = db_name if db_name else self.db_name
        new_binary = binary if binary else self.binary
        sub_committee = BenchmarkCommittee(new_organizer, new_db_name, new_binary, load_past_data=False)

        for node in self.G.nodes:
            if include_agents and exclude_agents:
                assert not (node in include_agents and node in exclude_agents)
            if exclude_agents and node in exclude_agents:
                continue
            if not include_agents or node in include_agents:
                ix, is_new_node = sub_committee._add_agent_node(node)
                assert is_new_node
                assert sub_committee.W_matrix.shape[0] == ix + 1, f'{sub_committee.W_matrix.shape[0]} != {ix + 1}'

        for edge in self.G.edges:
            if exclude_agents and (edge[0] in exclude_agents or edge[1] in exclude_agents):
                continue

            if exclude_edges and (edge[0], edge[1]) in exclude_edges:
                continue

            if not include_agents or (edge[0] in include_agents and edge[1] in include_agents):
                ix1 = self.G.nodes[edge[0]]['ix']
                ix2 = self.G.nodes[edge[1]]['ix']
                sub_ix1 = sub_committee.G.nodes[edge[0]]['ix']
                sub_ix2 = sub_committee.G.nodes[edge[1]]['ix']
                sub_committee.G.add_edge(edge[0], edge[1], n_games=self.W_matrix[ix1, ix2] + self.W_matrix[ix2, ix1])
                sub_committee.W_matrix[sub_ix1, sub_ix2] = self.W_matrix[ix1, ix2]
                sub_committee.W_matrix[sub_ix2, sub_ix1] = self.W_matrix[ix2, ix1]

        return sub_committee

    def _add_agent_node(self, agent: Agent) -> int:
        if agent not in self.G.nodes:
            ix = len(self.G.nodes)
            self.G.add_node(agent, ix=ix)
            self._expand_matrix()
            is_new_node = True
        else:
            ix = self.G.nodes[agent]['ix']
            is_new_node = False
        return ix, is_new_node

    def _expand_matrix(self):
        n = self.W_matrix.shape[0]
        new_matrix = np.zeros((n + 1, n + 1), dtype=float)
        new_matrix[:n, :n] = self.W_matrix
        self.W_matrix = new_matrix

