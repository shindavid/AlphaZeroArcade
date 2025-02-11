from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings, extract_match_record
from alphazero.logic import constants
from util.str_util import make_args_str
from util import subprocess_util
from util.logging_util import get_logger
from util.sqlite3_util import DatabaseConnectionPool, copy_db

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm

from itertools import combinations, product
from typing import List, Dict, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import os
import random

logger = get_logger()

@dataclass(frozen=True)
class MCTSAgent:
    gen: int = 0
    n_iters: int = 0

    def __repr__(self):
        return f'{self.gen}-{self.n_iters}'

@dataclass(frozen=True)
class PerfectAgent:
    strength: int = 0
    def __repr__(self):
        return f'Perfect-{self.strength}'

@dataclass(frozen=True)
class RandomAgent:
    def __repr__(self):
        return 'Random'

Agent = Union[MCTSAgent, PerfectAgent, RandomAgent]

@dataclass
class Match:
    agent1: Agent
    agent2: Agent
    n_games: int

@dataclass
class CommitteeData:
    W_matrix: np.ndarray
    G: nx.Graph

class DirectoryOrganizer:
    def __init__(self, game, tag, db_name='benchmark'):
        self.game = game
        self.tag = tag
        self.db_name = db_name

        self.base_dir = os.path.join('/workspace/repo')
        self.output_dir = os.path.join(self.base_dir, 'output', game, tag)
        self.database = os.path.join(self.output_dir, 'databases', f'{db_name}.db')
        self.binary = os.path.join(self.base_dir, 'target/Release/bin', game)
        self.model_dir = os.path.join(self.output_dir, 'models')

class BenchmarkCommittee:
    def __init__(self, organzier: DirectoryOrganizer, load_past_data: bool=False):
        self._organizer = organzier

        self.W_matrix = np.zeros((0, 0), dtype=float)
        self.G = nx.Graph()

        database_filepath = self._organizer.database
        self.benchmarking_db_conn_pool = DatabaseConnectionPool(database_filepath, \
            constants.BENCHMARKING_TABLE_CREATE_CMDS)
        self.binary = self._organizer.binary
        self.ratings = None
        if load_past_data:
            self.load_past_data()

    def expand_matrix(self):
        n = self.W_matrix.shape[0]
        new_matrix = np.zeros((n + 1, n + 1), dtype=float)
        new_matrix[:n, :n] = self.W_matrix
        self.W_matrix = new_matrix

    def load_past_data(self):
        conn = self.benchmarking_db_conn_pool.get_connection()
        c = conn.cursor()
        res = c.execute('SELECT gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws FROM matches')
        for gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws in res.fetchall():
            agent1 = BenchmarkCommittee.load_database_row(gen1, gen_iters1)
            agent2 = BenchmarkCommittee.load_database_row(gen2, gen_iters2)
            num_games = gen1_wins + gen2_wins + draws
            ix1, _ = self.add_agent_node(agent1)
            ix2, _ = self.add_agent_node(agent2)

            if self.G.has_edge(agent1, agent2):
                self.G.edges[(agent1, agent2)]['n_games'] += num_games
            else:
                self.G.add_edge(agent1, agent2, n_games=num_games)

            counts = WinLossDrawCounts(gen1_wins, gen2_wins, draws)
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def commit_counts(self, agent1: Agent, agent2: Agent, record: WinLossDrawCounts):
        conn = self.benchmarking_db_conn_pool.get_connection()
        gen1, n_iters1 = BenchmarkCommittee.make_database_row(agent1)
        gen2, n_iters2 = BenchmarkCommittee.make_database_row(agent2)
        match_tuple = (gen1, gen2, n_iters1, n_iters2, record.win, record.loss, record.draw)
        c = conn.cursor()
        c.execute('INSERT INTO matches (gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws) \
                  VALUES (?, ?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def add_agent_node(self, agent: MCTSAgent):
        if agent not in self.G.nodes:
            ix = len(self.G.nodes)
            self.G.add_node(agent, ix=ix)
            self.expand_matrix()
            is_new_node = True
        else:
            ix = self.G.nodes[agent]['ix']
            is_new_node = False
        return ix, is_new_node

    @staticmethod
    def gen_matches_from_latest(latest_gen: int, n_iters: int, n_games: int):
        gen_matches = BenchmarkCommittee.get_anchor_matches(latest_gen)
        matches = []
        for gen1, gen2 in gen_matches:
            if gen1 == 0:
                agent1 = RandomAgent()
            else:
                agent1 = MCTSAgent(gen=gen1, n_iters=n_iters)

            if gen2 == 0:
                agent2 = RandomAgent()
            else:
                agent2 = MCTSAgent(gen=gen2, n_iters=n_iters)
            match = Match(agent1=agent1, agent2=agent2, n_games=n_games)
            matches.append(match)
        return matches

    @staticmethod
    def get_matches(gen_start, gen_end, n_iters=100, freq=1, n_games=100):
        gens = range(gen_start, gen_end + 1, freq)
        matches = []
        for gen1, gen2 in combinations(gens, 2):
            if gen1 == 0:
                agent1 = RandomAgent()
            else:
                agent1 = MCTSAgent(gen=gen1, n_iters=n_iters)

            if gen2 == 0:
                agent2 = RandomAgent()
            else:
                agent2 = MCTSAgent(gen=gen2, n_iters=n_iters)
            match = Match(agent1=agent1, agent2=agent2, n_games=n_games)
            matches.append(match)
        return matches

    def play_matches(self, matches: List[Match], additional=False):
      for match in tqdm(matches):
        ix1, is_new_node1 = self.add_agent_node(match.agent1)
        ix2, is_new_node2 = self.add_agent_node(match.agent2)

        if self.G.has_edge(match.agent1, match.agent2):
          if not additional:
            n_games_played = self.G.edges[(match.agent1, match.agent2)]['n_games']
            match.n_games = match.n_games - n_games_played
          if match.n_games < 1:
            continue
        else:
          self.G.add_edge(match.agent1, match.agent2, n_games=0)

        result = self.run_match_helper(match, self.binary)
        self.W_matrix[ix1, ix2] += result.win + 0.5 * result.draw
        self.W_matrix[ix2, ix1] += result.loss + 0.5 * result.draw
        self.G[match.agent1][match.agent2]['n_games'] += match.n_games
        self.commit_counts(match.agent1, match.agent2, result)

    def compute_ratings(self, eps=1e-5):
        # Laplace smoothing for disconnected vertices
        W = self.W_matrix + eps
        np.fill_diagonal(W, 0)

        ratings = compute_ratings(W).tolist()
        self.ratings = {agent: ratings[ix] for agent, ix in self.G.nodes(data='ix')}

    def sub_committee(self, include_agents: List[Agent]=None, exclude_agents: List[Agent]=None, \
        exclude_edges: List[Tuple[Agent, Agent]]=None, organizer: DirectoryOrganizer=None)\
            -> 'BenchmarkCommittee':
        if organizer:
            sub_committee = BenchmarkCommittee(organizer, load_past_data=False)
        else:
            sub_committee = BenchmarkCommittee(self._organizer, load_past_data=False)


        for node in self.G.nodes:
            if include_agents and exclude_agents:
                assert not (node in include_agents and node in exclude_agents)
            if exclude_agents and node in exclude_agents:
                continue
            if not include_agents or node in include_agents:
                ix, is_new_node = sub_committee.add_agent_node(node)
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

    @staticmethod
    def get_anchor_numbers(gen: int) -> List[int]:
        return np.power(2, np.log2(gen).astype(int)) - np.power(2, np.arange(np.log2(gen).astype(int) + 1))

    @staticmethod
    def get_anchor_gens(gen: int) -> List[int]:
        log = np.log2(gen).astype(int)
        gen_dist = gen - np.power(2, log)
        close_gens = np.array([])
        if gen_dist > 0:
            close_gens = gen - np.power(2, np.arange(np.log2(gen_dist).astype(int) + 1))
        gens = np.concatenate([np.array([gen]), close_gens, BenchmarkCommittee.get_anchor_numbers(gen)]).astype(int).tolist()
        return sorted(gens)

    @staticmethod
    def get_anchor_matches(gen: int) -> List[List[int]]:
        matches = []
        for i in range(1, np.log2(gen).astype(int) + 1):
            a = BenchmarkCommittee.get_anchor_gens(np.power(2, i).astype(int))
            m = list(combinations(a, 2))
            matches += m
        a = BenchmarkCommittee.get_anchor_gens(gen)
        m = list(combinations(a, 2))
        matches += m
        return np.unique(np.array(matches), axis=0).astype(int).tolist()

    def get_model_path(self, gen: int) -> str:
        return os.path.join(self._organizer.model_dir, f'gen-{gen}.pt')

    def get_mcts_player_str(self, agent: MCTSAgent):
        gen = agent.gen
        n_iters = agent.n_iters
        player_args = {
            '--type': 'MCTS-C',
            '--name': str(agent),
            '-i': n_iters,
            '-m': self.get_model_path(gen),
            '-n': 1,
        }
        return make_args_str(player_args)

    @staticmethod
    def get_random_player_str():
        player_args = {
            '--type': 'MCTS-C',
            '--name': 'Random',
            '-i': 0,
            '-n': 1,
            '--no-model': None,
        }
        return make_args_str(player_args)

    @staticmethod
    def get_reference_player_str(agent: PerfectAgent):
        strength = agent.strength
        player_args = {
            '--type': 'Perfect',
            '--name': str(agent),
            '--strength': strength,
        }
        return make_args_str(player_args)

    def get_player_str(self, agent: Agent):
        if isinstance(agent, MCTSAgent):
            return self.get_mcts_player_str(agent)
        elif isinstance(agent, PerfectAgent):
            return BenchmarkCommittee.get_reference_player_str(agent)
        elif isinstance(agent, RandomAgent):
            return BenchmarkCommittee.get_random_player_str()

    @staticmethod
    def make_database_row(agent: Agent):
        if isinstance(agent, MCTSAgent):
            return agent.gen, agent.n_iters
        elif isinstance(agent, PerfectAgent):
            return -1, agent.strength
        elif isinstance(agent, RandomAgent):
            return 0, 0

    @staticmethod
    def load_database_row(gen, n_iters):
        if gen == -1:
            return PerfectAgent(strength=n_iters)
        elif gen == 0:
            return RandomAgent()
        else:
            return MCTSAgent(gen=gen, n_iters=n_iters)

    def run_match_helper(self, match: Match, binary):
        agent1 = match.agent1
        agent2 = match.agent2
        n_games = match.n_games
        if n_games < 1:
            return WinLossDrawCounts()

        ps1 = self.get_player_str(agent1)
        ps2 = self.get_player_str(agent2)

        base_args = {
            '-G': n_games,
            '--do-not-report-metrics': None,
        }

        args1 = dict(base_args)
        args2 = dict(base_args)

        port = 1234  # TODO: move this to constants.py or somewhere

        cmd1 = [
            binary,
            '--port', str(port),
            '--player', f'"{ps1}"',
        ]
        cmd1.append(make_args_str(args1))
        cmd1 = ' '.join(map(str, cmd1))

        cmd2 = [
            binary,
            '--remote-port', str(port),
            '--player', f'"{ps2}"',
        ]
        cmd2.append(make_args_str(args2))
        cmd2 = ' '.join(map(str, cmd2))

        proc1 = subprocess_util.Popen(cmd1)
        proc2 = subprocess_util.Popen(cmd2)

        expected_rc = None
        print_fn = logger.error
        stdout = subprocess_util.wait_for(proc1, expected_return_code=expected_rc, print_fn=print_fn)

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info('Match result: %s', record.get(0))
        return record.get(0)

class Evaluation:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_committee: BenchmarkCommittee):
        self._organizer = organizer
        self.benchmark_committee = benchmark_committee
        assert self.benchmark_committee.ratings is not None
        self.evaluation_db_conn_pool = DatabaseConnectionPool(self._organizer.database, \
            constants.BENCHMARKING_TABLE_CREATE_CMDS)
        self.eval = self.benchmark_committee.sub_committee(organizer=self._organizer)
        self.benchmark_ratings = dict(sorted(self.benchmark_committee.ratings.items(), key=lambda x: x[1]))

    def select_representative_agents(self, n_agents: int):
        percentile_groups = {i: list(group) for i, group in enumerate(np.array_split(list(self.benchmark_ratings.keys()), n_agents))}
        return [random.choice(group) for group in percentile_groups.values()]

    def evalute_against_benchmark(self, test_agent: Agent, n_games: int):
        representatives = self.select_representative_agents(10)
        matches = []
        for rep in representatives:
            matches.append(Match(test_agent, rep, n_games))
        self.eval.play_matches(matches, additional=True)
        eval_sub_committee = self.eval.sub_committee(include_agents=[test_agent] + representatives)
        eval_sub_committee.compute_ratings()

        interp_table = {v: self.benchmark_committee.ratings[k] for k, v in eval_sub_committee.ratings.items() if k != test_agent}
        interp_table = sorted(interp_table.items(), key=lambda x: x[1])
        interp_table = list(zip(*interp_table))
        x_values = interp_table[0]
        y_values = interp_table[1]
        x = eval_sub_committee.ratings[test_agent]
        test_rating = np.interp(x, x_values, y_values)

        return test_rating

def save_ratings_plt(agents, ratings, filename: str):
    agent_labels = np.array(list(agents)).astype(str)
    plt.figure()
    plt.scatter(agent_labels, ratings, label='Rating')
    plt.xticks(agent_labels[::5])
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    plt.savefig(filename)

if __name__ == '__main__':
    game = 'c4'
    tag = 'benchmark'
    n_games = 100
    organzier = DirectoryOrganizer(game, tag, db_name='evaluation')
    committee = BenchmarkCommittee(organzier, load_past_data=True)
    matches = BenchmarkCommittee.get_matches(0, 128, n_iters=100, freq=4, n_games=n_games)
    committee.play_matches(matches)
    committee.compute_ratings()

    evaluation = Evaluation(organzier, committee)