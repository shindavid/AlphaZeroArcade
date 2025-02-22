from alphazero.logic import constants
from alphazero.logic.agent_types import Agent, MCTSAgent, ReferenceAgent
from alphazero.logic.ratings import WinLossDrawCounts
from util.sqlite3_util import DatabaseConnectionPool

import numpy as np

from dataclasses import dataclass
from typing import Iterator, Tuple, Dict


@dataclass
class AgentEntry:
    gen: int
    n_iters: int
    set_temp_zero: bool
    binary_filename: str
    model_filename: str


@dataclass
class Entry:
    gen1: int
    gen2: int
    gen_iters1: int
    gen_iters2: int
    gen1_wins: int
    gen2_wins: int
    draws: int
    binary1: str
    binary2: str
    model_file1: str
    model_file2: str
    is_zero_temp1: bool
    is_zero_temp2: bool


class RatingDB:
    def __init__(self, db_filename: str):
        self.db_filename = db_filename
        self.db_conn_pool = DatabaseConnectionPool(db_filename,
                                                   constants.ARENA_TABLE_CREATE_CMDS)

    @staticmethod
    def entry_to_agent_entries(entry: Entry) -> Tuple[AgentEntry, AgentEntry]:
        agent1 = AgentEntry(entry.gen1,
                            entry.gen_iters1,
                            entry.is_zero_temp1,
                            entry.binary1,
                            entry.model_file1)
        agent2 = AgentEntry(entry.gen2,
                            entry.gen_iters2,
                            entry.is_zero_temp2,
                            entry.binary2,
                            entry.model_file2)
        return agent1, agent2

    @staticmethod
    def entry_to_counts(entry: Entry) -> WinLossDrawCounts:
        return WinLossDrawCounts(entry.gen1_wins, entry.gen2_wins, entry.draws)

    @staticmethod
    def build_agents_from_entry(entry: AgentEntry) -> Agent:
        if entry.gen == -1:
            type_str, strength_param = entry.model_filename.split('-')
            strength = entry.n_iters
            binary_filename = entry.binary_filename
            return ReferenceAgent(type_str,
                                  strength_param,
                                  strength,
                                  binary_filename)
        else:
            return MCTSAgent(entry.gen,
                             entry.n_iters,
                             entry.set_temp_zero,
                             entry.binary_filename,
                             entry.model_filename)

    @staticmethod
    def get_entry_from_agent(agent: Agent):
        if isinstance(agent, MCTSAgent):
            gen = agent.gen
            n_iters = agent.n_iters
            set_temp_zero = agent.set_temp_zero
            binary_filename = agent.binary_filename
            model_filename = agent.model_filename
        elif isinstance(agent, ReferenceAgent):
            gen = -1
            n_iters = agent.strength
            set_temp_zero = None
            binary_filename = agent.binary_filename
            model_filename = f'{agent.type_str}-{agent.strength_param}'
        return AgentEntry(gen, n_iters, set_temp_zero, binary_filename, model_filename)

    def fetchall(self) -> Iterator[Tuple[Agent, Agent, WinLossDrawCounts]]:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT gen1, gen2, gen_iters1, gen_iters2, \
            gen1_wins, gen2_wins, draws, binary1, binary2, model_file1, model_file2, \
                is_zero_temp1, is_zero_temp2 FROM matches')
        for row in c:
            agent_entry1, agent_entry2 = RatingDB.entry_to_agent_entries(Entry(*row))
            agent1 = RatingDB.build_agents_from_entry(agent_entry1)
            agent2 = RatingDB.build_agents_from_entry(agent_entry2)
            counts = RatingDB.entry_to_counts(Entry(*row))
            yield agent1, agent2, counts


    def commit_counts(self, agent1: Agent, agent2: Agent, record: WinLossDrawCounts):
        conn = self.db_conn_pool.get_connection()
        entry1: AgentEntry = RatingDB.get_entry_from_agent(agent1)
        entry2: AgentEntry = RatingDB.get_entry_from_agent(agent2)
        match_tuple = (entry1.gen,
                    entry2.gen,
                    entry1.n_iters,
                    entry2.n_iters,
                    record.win,
                    record.loss,
                    record.draw,
                    entry1.binary_filename,
                    entry2.binary_filename,
                    entry1.model_filename,
                    entry2.model_filename,
                    entry1.set_temp_zero,
                    entry2.set_temp_zero)
        c = conn.cursor()
        c.execute('INSERT INTO matches \
            (gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws, \
                binary1, binary2, model_file1, model_file2, is_zero_temp1, is_zero_temp2) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def commit_rating(self, agent, rating, benchmark_agents=None, benchmark_tag=None):
        conn = self.db_conn_pool.get_connection()
        agent_entry = RatingDB.get_entry_from_agent(agent)
        if benchmark_agents:
            benchmark_agents_str = ', '.join([str(a) for a in benchmark_agents])
        else:
            benchmark_agents_str = None

        entry_tuple = (agent_entry.gen,
                    agent_entry.n_iters,
                    rating,
                    agent_entry.binary_filename,
                    agent_entry.model_filename,
                    agent_entry.set_temp_zero,
                    benchmark_tag,
                    benchmark_agents_str)
        c = conn.cursor()
        c.execute('INSERT INTO ratings (gen, n_iters, rating, binary, model_file, \
            is_zero_temp, benchmark_tag, benchmark_agents) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)',  entry_tuple)
        conn.commit()

    def load_ratings(self) -> Dict[Agent, float]:
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT gen, n_iters, rating, binary, model_file, is_zero_temp, \
            benchmark_tag, benchmark_agents FROM ratings')
        ratings = {}
        for gen, n_iters, rating, binary, model_file, set_temp_zero, _, _ in c.fetchall():
            agent_entry = AgentEntry(gen, n_iters, set_temp_zero, binary, model_file)
            agent = RatingDB.build_agents_from_entry(agent_entry)
            ratings[agent] = rating
        return ratings


