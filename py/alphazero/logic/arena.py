from alphazero.logic.agent_types import Agent, AgentDBId, AgentRole, IndexedAgent, MatchType
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings
from alphazero.logic.rating_db import DBAgentRating, RatingDB
from util.index_set import IndexSet

import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class RatingData:
    agent_ids: np.ndarray
    ratings: np.ndarray
    committee: IndexSet
    tag: str


class Arena:
    """
    Arena is a data structure for storing the match results of agents. It provides functions to
    load match results from a database, play matches, commit results to a database,
    compute ratings, and create a subset of match results.
    """
    def __init__(self):
        self._W_matrix = np.zeros((0, 0), dtype=float)
        self._indexed_agents: List[IndexedAgent] = []
        self._agent_lookup: Dict[Agent, IndexedAgent] = {}
        self._agent_lookup_db_id: Dict[AgentDBId, IndexedAgent] = {}
        self._ratings: np.ndarray = np.array([])

    def load_agents_from_db(self, db: RatingDB, role: Optional[AgentRole] = None):
        for db_agent in db.fetch_agents():
            if role not in db_agent.roles:
                continue
            self.add_agent(db_agent.agent, roles=db_agent.roles, db_id=db_agent.db_id,
                           expand_matrix=False)
        self._expand_matrix()

    def load_matches_from_db(self, db: RatingDB, type: Optional[MatchType] = None) -> List[Agent]:
        for result in db.fetch_match_results():
            if type is not None and result.type != type:
                continue
            ix1 = self._agent_lookup_db_id[result.agent_id1].index
            ix2 = self._agent_lookup_db_id[result.agent_id2].index
            counts = result.counts

            self._W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self._W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def play_matches(self, matches: List[Match], binary: str, db: Optional[RatingDB] = None)\
            -> WinLossDrawCounts:
        """
        Play matches between agents and update the W_matrix with the results. If db is provided,
        the results are committed to the database.
        """
        counts_list = []
        for match in matches:
            indexed_agent1 = self._agent_lookup.get(match.agent1, None)
            indexed_agent2 = self._agent_lookup.get(match.agent2, None)

            ix1 = indexed_agent1.index
            ix2 = indexed_agent2.index

            if self._W_matrix[ix1, ix2] > 0 or self._W_matrix[ix2, ix1] > 0:
                n_games_played = int(self._W_matrix[ix1, ix2] + self._W_matrix[ix2, ix1])
                n_games = match.n_games - n_games_played
                match = Match(match.agent1, match.agent2, n_games, match.type)
                if n_games < 1:
                    continue

            counts: WinLossDrawCounts = MatchRunner.run_match_helper(match, binary)
            self._W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self._W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw
            counts_list.append(counts)
            if db:
                db_id1 = indexed_agent1.db_id
                db_id2 = indexed_agent2.db_id
                db.commit_counts(db_id1, db_id2, counts, match.type)
        return counts_list

    def update_match_results(self, ix1: int, ix2: int, counts: WinLossDrawCounts, type: MatchType,
                             db: RatingDB):
        self._W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
        self._W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw
        db_id1 = self._indexed_agents[ix1].db_id
        db_id2 = self._indexed_agents[ix2].db_id
        db.commit_counts(db_id1, db_id2, counts, type)

    def refresh_ratings(self, eps=1e-3):
        """
        TODO: pass self._ratings to compute_ratings to speed up the rating computation.
        """
        if self._W_matrix.shape[0] == 0:
            return
        self._ratings = compute_ratings(self._W_matrix, eps=eps)

    def load_ratings_from_db(self, db: RatingDB, role: AgentRole) -> RatingData:
        db_agent_ratings: List[DBAgentRating] = db.load_ratings(role)
        db_ids = []
        ratings = []
        committee = []
        for db_agent_rating in db_agent_ratings:
            db_id = self._agent_lookup_db_id[db_agent_rating.agent_id].db_id
            db_ids.append(db_id)
            ratings.append(db_agent_rating.rating)
            committee.append(db_agent_rating.is_committee)

        tag = db_agent_ratings[0].rating_tag if db_agent_ratings else None
        db_ids = np.array(db_ids)
        ratings = np.array(ratings)
        committee = IndexSet.from_bits(np.array(committee, dtype=bool))
        return RatingData(db_ids, ratings, committee, tag)

    def num_matches(self) -> int:
        return np.sum(self._W_matrix)

    def get_past_opponents_ix(self, agent: Agent) -> np.ndarray:
        W = self._W_matrix
        ix = self._agent_lookup[agent].index
        return np.where((W[ix, :] > 0) | (W[:, ix] > 0))[0]

    def n_games_played(self, agent: Agent):
        ix = self._agent_lookup[agent].index
        return (np.sum(self._W_matrix[ix, :]) + np.sum(self._W_matrix[:, ix])).astype(int)

    def adjacent_matrix(self) -> np.ndarray:
        return (self._W_matrix > 0) | (self._W_matrix.T > 0)

    def add_agent(self, agent: Agent, roles: Set[AgentRole], db_id: Optional[AgentDBId] = None,
                  expand_matrix: bool = True, db: Optional[RatingDB] = None) -> IndexedAgent:
        """
        Between the two optional arguments db_id and db, exactly one of them must be provided.
        db_id is provided when the agent is already in the database and we want to load it.
        db is provided when the agent is new and we want to add it to the database.
        """
        iagent = self._agent_lookup.get(agent, None)

        if iagent is not None:
            iagent.roles.update(roles)
            if db and roles != iagent.roles:
                db.update_agent_roles(iagent)
            return iagent

        index = len(self._indexed_agents)
        iagent = IndexedAgent(agent=agent, index=index, roles=roles, db_id=db_id)
        self._indexed_agents.append(iagent)
        self._agent_lookup[agent] = iagent

        if expand_matrix:
            self._expand_matrix()
        if db:
            assert iagent.db_id is None
            db.commit_agent(iagent)

        assert iagent.db_id is not None
        self._agent_lookup_db_id[iagent.db_id] = iagent

        return iagent

    def _expand_matrix(self):
        n_old = self._W_matrix.shape[0]
        n_new = len(self._indexed_agents)
        if n_old == n_new:
            return

        ne_W_matrix = np.zeros((n_new, n_new), dtype=float)
        ne_W_matrix[:n_old, :n_old] = self._W_matrix
        self._W_matrix = ne_W_matrix

    """
    Do not modify the returned objects in the following properties.
    """
    @property
    def indexed_agents(self) -> List[IndexedAgent]:
        return self._indexed_agents

    @property
    def ratings(self) -> np.ndarray:
        return self._ratings

    @property
    def agent_lookup_db_id(self) -> Dict[AgentDBId, IndexedAgent]:
        return self._agent_lookup_db_id

    @property
    def agent_lookup(self) -> Dict[Agent, IndexedAgent]:
        return self._agent_lookup
