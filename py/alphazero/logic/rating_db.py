from alphazero.logic import constants
from alphazero.logic.agent_types import Agent, MCTSAgent, ReferenceAgent
from alphazero.logic.ratings import WinLossDrawCounts
from util.sqlite3_util import DatabaseConnectionPool

import numpy as np

from dataclasses import dataclass
from typing import Tuple, Dict, List, Iterable, Optional
import os


AgentDBId = int  # id in agents table of the database


@dataclass
class DBAgent:
    agent: Agent
    db_id: AgentDBId


@dataclass
class MatchResult:
    agent_id1: AgentDBId
    agent_id2: AgentDBId
    counts: WinLossDrawCounts


@dataclass
class DBAgentRating:
    agent_id: AgentDBId
    rating: float
    is_committee: bool


class RatingDB:
    def __init__(self, db_filename: str):
        self.db_filename = db_filename
        self.db_conn_pool = DatabaseConnectionPool(db_filename,
                                                   constants.ARENA_TABLE_CREATE_CMDS)

    def fetch_agents(self) -> Iterable[DBAgent]:
        """
        Constructs a DBAgent from each row of the agents table, and returns a list of them.

        TODO: potentially consider adding an optional argument to specify a minimum agent id to
        fetch. Callers could use this to fetch only agents added since the last fetch.
        """
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        query = '''SELECT agents.id, gen, n_iters, tag, is_zero_temp
                   FROM agents
                   JOIN mcts_agents
                   ON agents.sub_id = mcts_agents.id
                   WHERE subtype="mcts"
                   '''

        c.execute(query)
        for agent_id, gen, n_iters, tag, set_temp_zero in c.fetchall():
            agent = MCTSAgent(gen, n_iters, set_temp_zero, tag)
            yield DBAgent(agent, agent_id)

        query = '''SELECT agents.id, type_str, strength_param, strength, tag
                   FROM agents
                   JOIN ref_agents
                   ON agents.sub_id = ref_agents.id
                   WHERE subtype="ref"
                   '''

        c.execute(query)
        for agent_id, type_str, strength_param, strength, tag in c.fetchall():
            agent = ReferenceAgent(type_str, strength_param, strength, tag)
            yield DBAgent(agent, agent_id)

    def fetch_match_results(self) -> Iterable[MatchResult]:
        """
        Fetches rows from the matches table and creates Match objects from them.

        Returns an iterator over the newly-created matches. This means that if we call this method
        twice, the second time will return only those matches that were added to the database
        since the first call.
        """
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        query = '''SELECT id, agent_id1, agent_id2, agent1_wins, agent2_wins, draws
                     FROM matches
                  '''

        c.execute(query)
        for row in c:
            match_id, agent_id1, agent_id2, agent1_wins, agent2_wins, draws = row
            counts = WinLossDrawCounts(agent1_wins, agent2_wins, draws)
            match = MatchResult(agent_id1, agent_id2, counts)
            yield match

    def load_ratings(self) -> List[DBAgentRating]:

        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT agent_id, rating, is_committee FROM ratings')

        db_agent_ratings = []
        for agent_id, rating, is_committee in c.fetchall():
            db_agent_ratings.append(DBAgentRating(agent_id, rating, bool(is_committee)))
        return db_agent_ratings

    def commit_counts(self, agent_id1: int, agent_id2: int, record: WinLossDrawCounts):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        match_tuple = (agent_id1, agent_id2, record.win, record.loss, record.draw)
        c.execute('''INSERT INTO matches (agent_id1, agent_id2, agent1_wins, agent2_wins, draws)
                  VALUES (?, ?, ?, ?, ?)''', match_tuple)
        conn.commit()

    def commit_rating(self, agent_ids: List[AgentDBId], ratings: np.ndarray,
                      is_committee_flags: Optional[List[bool]]):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        if is_committee_flags is None:
            is_committee_flags = [None] * len(agent_ids)
        else:
            # sql requires ints, not bools
            is_committee_flags = [int(flag) for flag in is_committee_flags]

        rating_tuples = []
        for i, rating, is_committee in zip(agent_ids, ratings, is_committee_flags):
            rating_tuple = (i, rating, is_committee)
            rating_tuples.append(rating_tuple)
        c.executemany('''REPLACE INTO ratings (agent_id, rating, is_committee)
                      VALUES (?, ?, ?)''', rating_tuples)
        conn.commit()

    def commit_agent(self, agent: Agent) -> AgentDBId:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        if isinstance(agent, MCTSAgent):
            subtype = 'mcts'

            insert = '''INSERT INTO mcts_agents (gen, n_iters, tag, is_zero_temp)
                         VALUES (?, ?, ?, ?)'''
            c.execute(insert, (agent.gen, agent.n_iters, agent.tag, agent.set_temp_zero))
            conn.commit()
            sub_id = c.lastrowid
        elif isinstance(agent, ReferenceAgent):
            subtype = 'ref'

            insert = '''INSERT INTO ref_agents (type_str, strength_param, strength, tag)
                         VALUES (?, ?, ?, ?)'''
            c.execute(insert, (agent.type_str, agent.strength_param, agent.strength, agent.tag))
            conn.commit()
            sub_id = c.lastrowid
        else:
            raise ValueError(f'Unknown agent type: {type(agent)}')

        insert = '''INSERT INTO agents (sub_id, subtype) VALUES (?, ?)'''
        c.execute(insert, (sub_id, subtype))
        conn.commit()
        return c.lastrowid

