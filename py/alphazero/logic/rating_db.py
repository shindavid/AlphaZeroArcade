from alphazero.logic import constants
from alphazero.logic.agent_types import Agent, MCTSAgent, ReferenceAgent
from alphazero.logic.ratings import WinLossDrawCounts
from util.sqlite3_util import DatabaseConnectionPool

import numpy as np

from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, List
import os


@dataclass
class AgentEntry:
    ix: int
    gen: int
    n_iters: int
    set_temp_zero: bool
    binary_filename: str
    model_filename: str


class RatingDB:
    def __init__(self, db_filename: str):
        self.db_filename = db_filename
        db_exists = os.path.exists(db_filename)
        self.db_conn_pool = DatabaseConnectionPool(db_filename,
                                                   constants.ARENA_TABLE_CREATE_CMDS)
        if not db_exists:
            with self.db_conn_pool.get_connection() as conn:
                c = conn.cursor()
                for cmd in constants.ARENA_TABLE_CREATE_CMDS:
                    try:
                        c.execute(cmd)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                conn.commit()
            os.chmod(db_filename, 0o666)

    @staticmethod
    def build_agents_from_entry(entry: AgentEntry) -> Agent:
        if entry.n_iters == -1:
            type_str, strength_param = entry.model_filename.split('-')
            return ReferenceAgent(type_str=type_str,
                                  strength_param=strength_param,
                                  strength=entry.gen,
                                  binary_filename=entry.binary_filename)
        else:
            return MCTSAgent(entry.gen,
                             entry.n_iters,
                             entry.set_temp_zero,
                             entry.binary_filename,
                             entry.model_filename)

    @staticmethod
    def get_entry_from_agent(agent: Agent):
        if isinstance(agent, MCTSAgent):
            ix = agent.ix
            gen = agent.gen
            n_iters = agent.n_iters
            set_temp_zero = agent.set_temp_zero
            binary_filename = agent.binary_filename
            model_filename = agent.model_filename
        elif isinstance(agent, ReferenceAgent):
            ix = agent.ix
            gen = agent.strength
            n_iters = -1
            set_temp_zero = None
            binary_filename = agent.binary_filename
            model_filename = f'{agent.type_str}-{agent.strength_param}'
        return AgentEntry(ix, gen, n_iters, set_temp_zero, binary_filename, model_filename)

    def load_agents(self) -> Dict[int, Agent]:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT ix, gen, n_iters, binary, model_file, is_zero_temp FROM agents')

        agents = {}
        for ix, gen, n_iters, binary, model_file, set_temp_zero in c.fetchall():
            agent_entry = AgentEntry(ix, gen, n_iters, set_temp_zero, binary, model_file)
            agent = RatingDB.build_agents_from_entry(agent_entry)
            agent.ix = ix
            agents[ix] = agent
        return agents

    def fetch_all_matches(self) -> Iterator[Tuple[Agent, Agent, WinLossDrawCounts]]:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT ix1, ix2, ix1_wins, ix2_wins, draws FROM matches')
        for ix1, ix2, ix1_wins, ix2_wins, draws in c:
            counts = WinLossDrawCounts(ix1_wins, ix2_wins, draws)
            yield ix1, ix2, counts

    def load_ratings(self) -> Tuple[np.ndarray, np.ndarray]:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT ix, rating, is_committee FROM ratings')

        ratings = []
        committee_ix = []
        for ix, rating, is_committee in c.fetchall():
            ratings.append(rating)
            committee_ix.append(is_committee)
        return np.array(ratings), np.array(committee_ix)

    def commit_counts(self, ix1: int, ix2: int, record: WinLossDrawCounts):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        match_tuple = (ix1, ix2, record.win, record.loss, record.draw)
        c.execute('''INSERT INTO matches (ix1, ix2, ix1_wins, ix2_wins, draws)
                  VALUES (?, ?, ?, ?, ?)''', match_tuple)
        conn.commit()

    def commit_rating(self, ix: List[int], ratings: np.ndarray, is_committee_flags: List[str]):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        if is_committee_flags is None:
            is_committee_flags = [None] * len(agents)

        rating_tuples = []
        for i, rating, is_committee in zip(ix, ratings, is_committee_flags):
            rating_tuple = (i, rating, is_committee)
            rating_tuples.append(rating_tuple)
        c.executemany('''INSERT INTO ratings (ix, rating, is_committee)
                      VALUES (?, ?, ?)
                      ON CONFLICT(ix) DO UPDATE SET
                      rating = excluded.rating,
                      is_committee = excluded.is_committee''', rating_tuples)
        conn.commit()

    def commit_agent(self, agents: List[Agent]):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        agent_tuples = []
        for agent in agents:
            agent_entry = RatingDB.get_entry_from_agent(agent)
            agent_tuple = (agent.ix,
                        agent_entry.gen,
                        agent_entry.n_iters,
                        agent_entry.binary_filename,
                        agent_entry.model_filename,
                        agent_entry.set_temp_zero)
            agent_tuples.append(agent_tuple)

        c.executemany('''INSERT INTO agents (ix, gen, n_iters, binary, model_file, is_zero_temp)
                      VALUES (?, ?, ?, ?, ?, ?)''', agent_tuples)
        conn.commit()


