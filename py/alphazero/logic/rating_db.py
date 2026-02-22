from alphazero.logic import constants
from alphazero.logic.agent_types import Agent, AgentDBId, AgentRole, IndexedAgent, \
        MatchType, MCTSAgent, ReferenceAgent
from alphazero.logic.custom_types import RatingTag
from alphazero.logic.ratings import WinLossDrawCounts
from util.index_set import IndexSet
from util.sqlite3_util import DatabaseConnectionPool

import numpy as np

from dataclasses import dataclass
import json
from typing import Dict, List, Iterable, Optional, Set

Condition = str

@dataclass
class DBAgent:
    agent: Agent
    db_id: AgentDBId
    roles: Set[AgentRole]


@dataclass
class MatchResult:
    agent_id1: AgentDBId
    agent_id2: AgentDBId
    counts: WinLossDrawCounts
    type: MatchType
    rating_tag: RatingTag


@dataclass
class DBAgentRating:
    agent_id: AgentDBId
    rating: float
    is_committee: Optional[bool] = None
    agent_tag: Optional[str] = None
    level: Optional[str] = None
    rating_tag: str = ''


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

        query = '''SELECT agents.id, type_str, strength_param, strength, tag, role
                   FROM agents
                   JOIN ref_agents
                   ON agents.sub_id = ref_agents.id
                   WHERE subtype="ref"
                   '''

        c.execute(query)
        for agent_id, type_str, strength_param, strength, tag, roles in c.fetchall():
            agent = ReferenceAgent(type_str, strength_param, strength, tag)
            agent_roles = AgentRole.from_str(roles)
            yield DBAgent(agent, agent_id, agent_roles)

        columns = ['agents.id', 'gen', 'paradigm', 'n_iters', 'tag', 'is_zero_temp', 'role']

        query = '''SELECT %s
                   FROM agents
                   JOIN mcts_agents
                   ON agents.sub_id = mcts_agents.id
                   WHERE subtype="mcts"
                   ''' % ', '.join(columns)

        c.execute(query)
        for row in c.fetchall():
            agent_id, gen, paradigm, n_iters, tag, set_temp_zero, roles = row
            agent = MCTSAgent(paradigm, gen, n_iters, bool(set_temp_zero), tag)
            agent_roles = AgentRole.from_str(roles)
            yield DBAgent(agent, agent_id, agent_roles)

    def fetch_match_results(self, rating_tag: Optional[str] = None) -> Iterable[MatchResult]:
        if rating_tag is None:
            return self._fetch_match_results_table('')
        else:
            clause = f' WHERE rating_tag="{rating_tag}"'
            return self._fetch_match_results_table(clause)

    def _fetch_match_results_table(self, where_clause) -> Iterable[MatchResult]:
        """
        Fetches rows from the matches table and creates Match objects from them.

        Returns an iterator over the newly-created matches. This means that if we call this method
        twice, the second time will return only those matches that were added to the database
        since the first call.
        """
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        query = f'''SELECT id, agent_id1, agent_id2, agent1_wins, agent2_wins, draws, type, rating_tag
                     FROM matches{where_clause}
                  '''

        c.execute(query)
        for row in c:
            match_id, agent_id1, agent_id2, agent1_wins, agent2_wins, draws, type, rating_tag = row
            counts = WinLossDrawCounts(agent1_wins, agent2_wins, draws)
            match = MatchResult(agent_id1, agent_id2, counts, MatchType(type), rating_tag)
            yield match

    def load_ratings(self, role: AgentRole, rating_tag: Optional[str] = None) -> List[DBAgentRating]:
        if role == AgentRole.BENCHMARK:
            assert rating_tag is None, "rating_tag should be None when role is BENCHMARK"
            return self._load_benchmark_ratings_table()
        elif role == AgentRole.TEST:
            if rating_tag is None:
                return self._load_eval_ratings_table('')
            else:
                clause = f' AND evaluator_ratings.rating_tag="{rating_tag}"'
                return self._load_eval_ratings_table(clause)
        raise ValueError(f'Unknown role: {role}')

    def _load_benchmark_ratings_table(self) -> List[DBAgentRating]:
        db_agent_ratings = []
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        c.execute('''
            SELECT
                benchmark_ratings.agent_id,
                benchmark_ratings.rating,
                benchmark_ratings.is_committee,
                mcts_agents.tag,
                mcts_agents.gen
            FROM benchmark_ratings
            JOIN agents ON benchmark_ratings.agent_id = agents.id
            JOIN mcts_agents ON agents.subtype = 'mcts' AND agents.sub_id = mcts_agents.id
            WHERE agents.subtype = 'mcts';
            ''')
        for row in c.fetchall():
            agent_id, rating, is_committee, agent_tag, gen = row
            is_committee = bool(is_committee)
            db_agent_ratings.append(DBAgentRating(agent_id=agent_id,
                                                  rating=rating,
                                                  is_committee=is_committee,
                                                  agent_tag=agent_tag,
                                                  level=gen))

        c.execute('''
            SELECT
                benchmark_ratings.agent_id,
                benchmark_ratings.rating,
                benchmark_ratings.is_committee,
                ref_agents.tag,
                ref_agents.strength
            FROM benchmark_ratings
            JOIN agents ON benchmark_ratings.agent_id = agents.id
            JOIN ref_agents ON agents.subtype = 'ref' AND agents.sub_id = ref_agents.id
            WHERE agents.subtype = 'ref';
            ''')
        for row in c.fetchall():
            agent_id, rating, is_committee, agent_tag, strength = row
            is_committee = bool(is_committee)
            db_agent_ratings.append(DBAgentRating(agent_id=agent_id,
                                                  rating=rating,
                                                  is_committee=is_committee,
                                                  agent_tag=agent_tag,
                                                  level=strength))
        return db_agent_ratings

    def _load_eval_ratings_table(self, additional_where_clause: Condition) -> List[DBAgentRating]:
        db_agent_ratings = []
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        c.execute(f'''
            SELECT
                evaluator_ratings.agent_id,
                evaluator_ratings.rating,
                evaluator_ratings.rating_tag,
                mcts_agents.tag,
                mcts_agents.gen
            FROM evaluator_ratings
            JOIN agents ON evaluator_ratings.agent_id = agents.id
            JOIN mcts_agents ON agents.subtype = 'mcts' AND agents.sub_id = mcts_agents.id
            WHERE agents.subtype = 'mcts'{additional_where_clause};
            ''')
        for row in c.fetchall():
            agent_id, rating, rating_tag, agent_tag, gen = row
            db_agent_ratings.append(DBAgentRating(agent_id=agent_id,
                                                rating=rating,
                                                agent_tag=agent_tag,
                                                level=gen,
                                                rating_tag=rating_tag))

        c.execute(f'''
            SELECT
                evaluator_ratings.agent_id,
                evaluator_ratings.rating,
                evaluator_ratings.rating_tag,
                ref_agents.tag,
                ref_agents.strength
            FROM evaluator_ratings
            JOIN agents ON evaluator_ratings.agent_id = agents.id
            JOIN ref_agents ON agents.subtype = 'ref' AND agents.sub_id = ref_agents.id
            WHERE agents.subtype = 'ref'{additional_where_clause};
            ''')
        for row in c.fetchall():
            agent_id, rating, rating_tag, agent_tag, strength = row
            db_agent_ratings.append(DBAgentRating(agent_id=agent_id,
                                                rating=rating,
                                                agent_tag=agent_tag,
                                                level=strength,
                                                rating_tag=rating_tag))
        return db_agent_ratings

    def commit_counts(self, agent_id1: int, agent_id2: int, record: WinLossDrawCounts,
                      type: MatchType, rating_tag: RatingTag = ''):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        match_tuple = (agent_id1, agent_id2, record.win, record.loss, record.draw, type.value, rating_tag)
        c.execute('''INSERT INTO matches (agent_id1, agent_id2, agent1_wins, agent2_wins, draws,
            type, rating_tag) VALUES (?, ?, ?, ?, ?, ?, ?)''', match_tuple)
        conn.commit()

    def commit_ratings(self, iagents: List[IndexedAgent], ratings: np.ndarray,
                       committee: Optional[IndexSet] = None, rating_tag: Optional[str] = None):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        benchmark_tuples = []
        evaluator_tuples = []
        for i in range(len(iagents)):
            iagent = iagents[i]
            rating = ratings[i]
            if AgentRole.BENCHMARK in iagent.roles and committee is not None:
                benchmark_tuples.append((iagent.db_id, rating, int(committee[i])))
            if AgentRole.TEST in iagent.roles:
                evaluator_tuples.append((iagent.db_id, rating, rating_tag))

        c.executemany('''REPLACE INTO benchmark_ratings (agent_id, rating, is_committee)
                      VALUES (?, ?, ?)''', benchmark_tuples)
        c.executemany('''REPLACE INTO evaluator_ratings (agent_id, rating, rating_tag)
                      VALUES (?, ?, ?)''', evaluator_tuples)
        conn.commit()

    def commit_agent(self, iagent: IndexedAgent):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        agent = iagent.agent
        if isinstance(agent, MCTSAgent):
            subtype = 'mcts'

            insert = '''INSERT INTO mcts_agents (paradigm, gen, n_iters, tag, is_zero_temp)
                         VALUES (?, ?, ?, ?, ?)'''
            c.execute(insert, (agent.paradigm, agent.gen, agent.n_iters, agent.tag,
                               agent.set_temp_zero))
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

        insert = '''INSERT INTO agents (sub_id, subtype, role) VALUES (?, ?, ?)'''
        agent_roles = AgentRole.to_str(iagent.roles)
        c.execute(insert, (sub_id, subtype, agent_roles))
        conn.commit()
        iagent.db_id = c.lastrowid

    def update_agent_roles(self, iagent: IndexedAgent):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()

        agent_roles = AgentRole.to_str(iagent.roles)
        c.execute('''UPDATE agents SET role=? WHERE id=?''', (agent_roles, iagent.db_id))
        conn.commit()

    def is_empty(self):
        conn = self.db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('SELECT 1 FROM agents LIMIT 1')
        return c.fetchone() is None

    @staticmethod
    def save_ratings_to_json(iagents: List[IndexedAgent], ratings: np.ndarray, file: str,
                             cmd_used: str):
        data = {}
        data['cmd_used'] = json.dumps(cmd_used)[1:-1]  # Remove quotes around the command string
        for ia, elo in zip(iagents, ratings):
            data[str(ia.agent)] = {
                'iagent': ia.to_dict(),
                'rating': elo
            }

        with open(file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_ratings_from_json(self, json_file: str):
        with open(json_file, 'r') as f:
            data: Dict = json.load(f)

        iagents = []
        ratings = []
        for key, entry in data.items():
            if key == 'cmd_used':
                continue
            agent: Agent = None
            iagent_dict = entry['iagent']
            if iagent_dict['agent']['type'] == 'MCTS':
                agent = MCTSAgent(**iagent_dict['agent']['data'])
            elif iagent_dict['agent']['type'] == 'Reference':
                agent = ReferenceAgent(**iagent_dict['agent']['data'])
            else:
                raise ValueError(f"unknown agent type {iagent_dict['agent']['type']}")

            ia = IndexedAgent(agent=agent,
                              index=iagent_dict['index'],
                              roles=AgentRole.from_str(iagent_dict['roles']),
                              db_id=iagent_dict['db_id'])

            self.commit_agent(ia)
            iagents.append(ia)
            ratings.append(entry['rating'])

        self.commit_ratings(iagents, ratings,
                            committee=IndexSet.from_bits(np.ones(len(iagents), dtype=bool)))

    @property
    def db_lock(self):
        return self.db_conn_pool._db_lock
