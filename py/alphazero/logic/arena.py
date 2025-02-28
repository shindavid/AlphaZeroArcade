from alphazero.logic.agent_types import Agent
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings
from alphazero.logic.rating_db import AgentDBId, DBAgent, RatingDB, DBAgentRating

from dataclasses import replace
import numpy as np

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


ArenaIndex = int  # index of an agent in an Arena


@dataclass
class IndexedAgent:
    """
    A dataclass for storing an agent with auxiliary info.

    - index refers to the index of the agent in the Arena's data structures.
    - db_id is the id of the agent in the database. This might be set after initial creation.
    """
    agent: Agent
    index: ArenaIndex
    db_id: Optional[AgentDBId] = None


@dataclass
class RatingArrays:
    ixs: np.ndarray
    ratings: np.ndarray
    committee_ixs: np.ndarray


class Arena:
    """
    Arena is a data structure for storing the match results of agents. It provides functions to
    load match results from a database, play matches, commit results to a database,
    compute ratings, and create a subset of match results.
    """
    def __init__(self):
        # TODO: consider making these members private, as there are sensitive invariants that
        # need to be maintained between them. Specifically, self.indexed_agents needs to be
        # consistent with the lookup dictionaries.
        self._W_matrix = np.zeros((0, 0), dtype=float)
        self.indexed_agents: List[IndexedAgent] = []
        self._agent_lookup: Dict[Agent, IndexedAgent] = {}
        self._agent_lookup_db_id: Dict[AgentDBId, IndexedAgent] = {}
        self.ratings: np.ndarray = np.array([])

    def load_agents_from_db(self, db: RatingDB):
        for db_agent in db.fetch_agents():
            self._add_agent(db_agent.agent, db_id=db_agent.db_id, expand_matrix=False)
        self._expand_matrix()

    def load_matches_from_db(self, db: RatingDB) -> List[Agent]:
        for result in db.fetch_match_results():
            ix1 = self._agent_lookup_db_id[result.agent_id1].index
            ix2 = self._agent_lookup_db_id[result.agent_id2].index
            counts = result.counts

            self._W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self._W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def play_matches(self, matches: List[Match], game: str, additional=False,
                     db: Optional[RatingDB]=None) -> WinLossDrawCounts:
        counts_list = []
        for match in matches:
            indexed_agent1 = self._add_agent(match.agent1, expand_matrix=True, db=db)
            indexed_agent2 = self._add_agent(match.agent2, expand_matrix=True, db=db)

            ix1 = indexed_agent1.index
            ix2 = indexed_agent2.index

            if self._W_matrix[ix1, ix2] > 0 or self._W_matrix[ix2, ix1] > 0:
                if not additional:
                    n_games_played = int(self._W_matrix[ix1, ix2] + self._W_matrix[ix2, ix1])
                    match.n_games = match.n_games - n_games_played
                if match.n_games < 1:
                    continue

            counts: WinLossDrawCounts = MatchRunner.run_match_helper(match, game)
            self._W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self._W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw
            counts_list.append(counts)
            if db:
                db_id1 = indexed_agent1.db_id
                db_id2 = indexed_agent2.db_id
                db.commit_counts(db_id1, db_id2, counts)
        return counts_list

    def commit_ratings_to_db(self, db: RatingDB, iagents: Iterable[IndexedAgent],
                             ratings: np.ndarray, is_committee_flags: List[bool] = None):
        agent_ids = [iagent.db_id for iagent in iagents]
        db.commit_rating(agent_ids, ratings, is_committee_flags=is_committee_flags)

    def compute_ratings(self, eps=1e-3) -> np.ndarray:
        self.ratings = compute_ratings(self._W_matrix, eps=eps)
        return self.ratings

    def load_ratings_from_db(self, db: RatingDB) -> Tuple[np.ndarray, np.ndarray]:
        db_agent_ratings: List[DBAgentRating] = db.load_ratings()
        ixs = []
        ratings = []
        committee_flags = []
        for db_agent_rating in db_agent_ratings:
            ix = self._agent_lookup_db_id[db_agent_rating.agent_id].index
            ixs.append(ix)
            ratings.append(db_agent_rating.rating)
            committee_flags.append(db_agent_rating.is_committee)
        ixs = np.array(ixs)
        ratings = np.array(ratings)
        committee_ixs = ixs[committee_flags]
        sorted_ixs = np.argsort(ixs)
        ixs = ixs[sorted_ixs]
        ratings = ratings[sorted_ixs]
        return RatingArrays(ixs, ratings, committee_ixs)

    def clone(self) -> 'Arena':
        new_arena = Arena()
        new_arena._W_matrix = self._W_matrix.copy()
        new_arena.indexed_agents = self.indexed_agents.copy() # shallow copy with same agents
        new_arena._agent_lookup = self._agent_lookup.copy()
        new_arena._agent_lookup_db_id = self._agent_lookup_db_id.copy()
        new_arena.ratings = self.ratings.copy()
        return new_arena

    def num_matches(self) -> int:
        return np.sum(self._W_matrix)

    def opponent_ix_played_against(self, agent: Agent) -> np.ndarray:
        ix = self._agent_lookup[agent].index
        vertical = np.where(self._W_matrix[:, ix] > 0)[0]
        horizontal = np.where(self._W_matrix[ix, :] > 0)[0]
        return np.union1d(vertical, horizontal)

    def n_games_played(self, agent: Agent):
        ix = self._agent_lookup[agent].index
        return (np.sum(self._W_matrix[ix, :]) + np.sum(self._W_matrix[:, ix])).astype(int)

    def adjacent_matrix(self) -> np.ndarray:
        return (self._W_matrix > 0) | (self._W_matrix.T > 0)

    def _add_agent(self, agent: Agent, db_id: Optional[AgentDBId]=None,
                   expand_matrix: bool=True, db: Optional[RatingDB]=None,
                   assert_new: bool = False) -> IndexedAgent:
        iagent = self._agent_lookup.get(agent, None)
        if assert_new:
            assert iagent is None

        if iagent is not None:
            return iagent

        index = len(self.indexed_agents)
        iagent = IndexedAgent(agent, index, db_id)
        self.indexed_agents.append(iagent)
        self._agent_lookup[agent] = iagent

        if expand_matrix:
            self._expand_matrix()
        if db:
            assert iagent.db_id is None
            iagent.db_id = db.commit_agent(agent)

        assert iagent.db_id is not None
        self._agent_lookup_db_id[iagent.db_id] = iagent

        return iagent

    def _expand_matrix(self):
        n_old = self._W_matrix.shape[0]
        n_new = len(self.indexed_agents)
        if n_old == n_new:
            return

        ne_W_matrix = np.zeros((n_new, n_new), dtype=float)
        ne_W_matrix[:n_old, :n_old] = self._W_matrix
        self._W_matrix = ne_W_matrix
