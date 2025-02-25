from alphazero.logic.agent_types import Agent
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings
from alphazero.logic.rating_db import RatingDB

from dataclasses import replace
import numpy as np

from typing import List, Dict, Tuple, Optional


class Arena:
    """
    Arena is a data structure for storing the match results of agents. It provides functions to
    load match results from a database, play matches, commit results to a database,
    compute ratings, and create a subset of match results.
    """
    def __init__(self):
        self.W_matrix = np.zeros((0, 0), dtype=float)
        self.agents: Dict[int, Agent] = {}
        self.ratings: np.ndarray = np.array([])

    def load_matches_from_db(self, database: RatingDB) -> List[Agent]:
        wld_dict = {}
        new_agents = []
        for agent1, agent2, counts in database.fetchall():
            ix1, is_new_agent1 = self._add_agent(agent1, expand_matrix=False)
            ix2, is_new_agent2 = self._add_agent(agent2, expand_matrix=False)
            wld_dict[(ix1, ix2)] = counts
            if is_new_agent1:
                new_agents.append(agent1)
            if is_new_agent2:
                new_agents.append(agent2)

        self._expand_matrix(len(self.agents))
        for (ix1, ix2), counts in wld_dict.items():
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

        return new_agents

    def play_matches(self, matches: List[Match], additional=False,
                     db: Optional[RatingDB]=None) -> WinLossDrawCounts:
        counts_list = []
        for match in matches:
            ix1, _ = self._add_agent(match.agent1, expand_matrix=True)
            ix2, _ = self._add_agent(match.agent2, expand_matrix=True)

            if self.W_matrix[ix1, ix2] > 0 or self.W_matrix[ix2, ix1] > 0:
                if not additional:
                    n_games_played = int(self.W_matrix[ix1, ix2] + self.W_matrix[ix2, ix1])
                    match.n_games = match.n_games - n_games_played
                if match.n_games < 1:
                    continue

            counts: WinLossDrawCounts = MatchRunner.run_match_helper(match)
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw
            counts_list.append(counts)
            if db:
                db.commit_counts(match.agent1, match.agent2, counts)
        return counts_list

    def commit_match_to_db(self, db_filename: str, match: Match, counts: WinLossDrawCounts):
        db.commit_counts(match.agent1, match.agent2, counts)

    def commit_ratings_to_db(self, db_filename: str, agents: List[Agent], ratings: np.ndarray,
                             is_committee_flags: List[str] = None):
        db = RatingDB(db_filename)
        db.commit_rating(agents, ratings, is_committee_flags=is_committee_flags)

    def compute_ratings(self, eps=1e-3) -> np.ndarray:
        self.ratings = compute_ratings(self.W_matrix, eps=eps)
        return self.ratings

    def load_ratings_from_db(self, db_filename: str) -> Dict[Agent, float]:
        db = RatingDB(db_filename)
        return db.load_ratings()

    def clone(self, include_agents: List[Agent] = None, exclude_agents: List[Agent] = None)\
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
        old_ix = {}
        for ix, agent in self.agents.items():
            if include_agents and exclude_agents:
                assert not (agent in include_agents and agent in exclude_agents)
            if exclude_agents and agent in exclude_agents:
                continue
            if not include_agents or agent in include_agents:
                agent_copy = replace(agent)
                new_ix, is_new_node = sub_arena._add_agent(agent_copy, expand_matrix=False)
                old_ix[new_ix] = ix
                assert is_new_node
        sub_arena._expand_matrix(len(sub_arena.agents))
        n = len(sub_arena.agents)
        for i in range(n):
            for j in range(n):
                sub_arena.W_matrix[i, j] = self.W_matrix[old_ix[i], old_ix[j]]
        sub_arena.compute_ratings()
        return sub_arena

    def opponent_ix_played_against(self, agent: Agent) -> np.ndarray:
        ix, is_new = self._add_agent[agent]
        assert not is_new
        vertical = np.where(self.W_matrix[:, ix] > 0)[0]
        horizontal = np.where(self.W_matrix[ix, :] > 0)[0]
        return np.union1d(vertical, horizontal)

    def n_games_played(self, agent: Agent):
        ix, is_new = self._add_agent[agent]
        assert not is_new
        return (np.sum(self.W_matrix[ix, :]) + np.sum(self.W_matrix[:, ix])).astype(int)

    def _add_agent(self, agent: Agent, expand_matrix: bool = True) -> Tuple[int, bool]:
        for ix, a in self.agents.items():
            if a == agent:
                return ix, False
        ix = len(self.agents)
        agent.ix = ix
        self.agents[ix] = agent
        if expand_matrix:
            self._expand_matrix(1)
        return ix, True

    def _expand_matrix(self, k: int):
        n = self.W_matrix.shape[0]
        new_matrix = np.zeros((n + k, n + k), dtype=float)
        new_matrix[:n, :n] = self.W_matrix
        self.W_matrix = new_matrix


