from alphazero.logic.agent_types import Agent
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings
from alphazero.logic.rating_db import RatingDB

from tqdm import tqdm
import numpy as np

from typing import List, Dict

class Arena:
    """
    Arena is a data structure for storing the match results of agents. It provides functions to
    load match results from a database, play matches, commit results to a database,
    compute ratings, and create a subset of match results.
    """
    def __init__(self):
        self.W_matrix = np.zeros((0, 0), dtype=float)
        self.agents_lookup: Dict[Agent, int] = {}

    def load_from_db(self, db_filename: str):
        db = RatingDB(db_filename)
        wld_dict = {}
        for agent1, agent2, counts in db.fetchall():
            ix1, _ = self._add_agent(agent1, expand_matrix=False)
            ix2, _ = self._add_agent(agent2, expand_matrix=False)
            wld_dict[(ix1, ix2)] = counts

        self._init_W_matrix(len(self.agents_lookup))
        counts: WinLossDrawCounts = WinLossDrawCounts()
        for (ix1, ix2), counts in wld_dict.items():
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def play_matches(self, matches: List[Match], additional=False) -> WinLossDrawCounts:
        iterator = tqdm(matches) if len(matches) > 1 else matches
        counts_list = []
        for match in iterator:
            ix1, _ = self._add_agent(match.agent1, expand_matrix=True)
            ix2, _ = self._add_agent(match.agent2, expand_matrix=True)

            if self.W_matrix[ix1, ix2] > 0 or self.W_matrix[ix2, ix1] > 0:
                if not additional:
                    n_games_played = self.W_matrix[ix1, ix2] + self.W_matrix[ix2, ix1]
                    match.n_games = match.n_games - n_games_played
                if match.n_games < 1:
                    continue

            counts: WinLossDrawCounts = MatchRunner.run_match_helper(match)
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw
            counts_list.append(counts)
        return counts_list

    def commit_match_to_db(self, db_filename: str, match: Match, counts: WinLossDrawCounts):
        db = RatingDB(db_filename)
        db.commit_counts(match.agent1, match.agent2, counts)

    def commit_ratings_to_db(self, db_filename: str, agents: List[Agent], ratings: np.ndarray):
        db = RatingDB(db_filename)
        for agent, rating in zip(agents, ratings):
            db.commit_rating(agent, rating)

    def compute_ratings(self, eps=1e-6) -> np.ndarray:
        ratings = compute_ratings(self.W_matrix, eps=eps)
        return ratings

    def create_subset(self, include_agents: List[Agent] = None, exclude_agents: List[Agent] = None)\
        -> 'Arena':
        """
        Create a new arena for a subset of agents.

        This method filters the current arena's agents based on the optional lists provided,
        then constructs a new arena with the filtered data. Any match results in W_matrix are
        copied over for agents that remain.

        Args:
            include_agents (List[Agent], optional):
                If provided, only these agents (and edges between them) are considered.
                Defaults to including all agents if not specified.
            exclude_agents (List[Agent], optional):
                If provided, these agents (and edges involving them) are excluded from
                the new committee.
        """

        sub_arena = Arena()

        for agent in self.agents_lookup:
            if include_agents and exclude_agents:
                assert not (agent in include_agents and agent in exclude_agents)
            if exclude_agents and agent in exclude_agents:
                continue
            if not include_agents or agent in include_agents:
                _, is_new_node = sub_arena._add_agent(agent, expand_matrix=False)
                assert is_new_node

        for agent_i, i in sub_arena.agents_lookup.items():
            for agent_j, j in sub_arena.agents_lookup.items():
                old_i = self.agents_lookup[agent_i]
                old_j = self.agents_lookup[agent_j]
                sub_arena.W_matrix[i, j] = self.W_matrix[old_i, old_j]

        return sub_arena

    def _add_agent(self, agent: Agent, expand_matrix: bool = True) -> int:
        if agent not in self.agents_lookup:
            ix = len(self.agents_lookup)
            self.agents_lookup[agent] = ix
            if expand_matrix:
                self._expand_matrix()
            is_new_node = True
        else:
            ix = self.agents_lookup[agent]
            is_new_node = False
        return ix, is_new_node

    def _init_W_matrix(self, n: int):
        self.W_matrix = np.zeros((n, n), dtype=float)

    def _expand_matrix(self):
        n = self.W_matrix.shape[0]
        new_matrix = np.zeros((n + 1, n + 1), dtype=float)
        new_matrix[:n, :n] = self.W_matrix
        self.W_matrix = new_matrix