from alphazero.logic import constants
from alphazero.logic.agent_types import Agent, PerfectAgent, UniformAgent, MCTSAgent
from alphazero.logic.ratings import WinLossDrawCounts
from util.sqlite3_util import DatabaseConnectionPool

from typing import List
import os

class RatingDB:
    def __init__(self, db_dir: str, db_name: str=None):
        self.db_dir = db_dir
        self.db_name = db_name
        db_path = os.path.join(db_dir, db_name + '.db')
        self.db_conn_pool = DatabaseConnectionPool(db_path, constants.BENCHMARKING_TABLE_CREATE_CMDS)

    @staticmethod
    def build_agent_from_row(gen, n_iters, organizer: str=None) -> Agent:
        if gen == -1:
            return PerfectAgent(strength=n_iters)
        elif gen == 0:
            return UniformAgent(n_iters=n_iters)
        else:
            return MCTSAgent(gen=gen, n_iters=n_iters, organizer=organizer)

    @staticmethod
    def get_gen_iter_from_agent(agent: Agent):
        if isinstance(agent, MCTSAgent):
            return agent.gen, agent.n_iters
        elif isinstance(agent, PerfectAgent):
            return -1, agent.strength
        elif isinstance(agent, UniformAgent):
            return 0, agent.n_iters

    def fetchall(self) -> List:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        res = c.execute('SELECT gen1, gen2, gen_iters1, gen_iters2, \
          gen1_wins, gen2_wins, draws FROM matches')
        rows = res.fetchall()
        return rows

    def commit_counts(self, agent1: Agent, agent2: Agent, record: WinLossDrawCounts):
        conn = self.db_conn_pool.get_connection()
        gen1, n_iters1 = RatingDB.get_gen_iter_from_agent(agent1)
        gen2, n_iters2 = RatingDB.get_gen_iter_from_agent(agent2)
        match_tuple = (gen1, gen2, n_iters1, n_iters2, record.win, record.loss, record.draw)
        c = conn.cursor()
        c.execute('INSERT INTO matches (gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws) \
                  VALUES (?, ?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def commit_rating(self, agent, rating, benchmark_agents, benchmark_tag):
        conn = self.db_conn_pool.get_connection()
        gen, n_iters = RatingDB.get_gen_iter_from_agent(agent)
        benchmark_agents_str = ', '.join([str(a) for a in benchmark_agents])
        match_tuple = (gen, n_iters, rating, benchmark_tag, benchmark_agents_str)
        c = conn.cursor()
        c.execute('INSERT INTO ratings (gen, n_iters, rating, benchmark_tag, benchmark_agents) \
                  VALUES (?, ?, ?, ?, ?)', match_tuple)
        conn.commit()
