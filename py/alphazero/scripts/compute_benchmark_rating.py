from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings, extract_match_record
from alphazero.logic import constants
from util.str_util import make_args_str
from util import subprocess_util
from util.logging_util import get_logger
from util.sqlite3_util import DatabaseConnectionPool

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm

from itertools import combinations
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

game = 'c4'
tag = 'benchmark'
binary = f'/workspace/repo/target/Release/bin/{game}'
database = f'/workspace/repo/output/{game}/{tag}/benchmarking.db'
logger = get_logger()

@dataclass(frozen=True)
class Agent:
    gen: int
    n_iters: int

    def __repr__(self):
        return f'{self.gen}-{self.n_iters}'

def get_anchor_numbers(gen: int) -> List[int]:
    return np.power(2, np.log2(gen).astype(int)) - np.power(2, np.arange(np.log2(gen).astype(int) + 1))

def get_anchor_gens(gen: int) -> List[int]:
    log = np.log2(gen).astype(int)
    gen_dist = gen - np.power(2, log)
    close_gens = np.array([])
    if gen_dist > 0:
        close_gens = gen - np.power(2, np.arange(np.log2(gen_dist).astype(int) + 1))
    gens = np.concatenate([np.array([gen]), close_gens, get_anchor_numbers(gen)])
    return sorted(gens)

def get_matches(gen: int) -> List[int]:
    matches = []
    for i in range(1, np.log2(gen).astype(int) + 1):
      a = get_anchor_gens(np.power(2, i).astype(int))
      m = list(combinations(a, 2))
      matches += m
    a = get_anchor_gens(gen)
    m = list(combinations(a, 2))
    matches += m
    return np.unique(np.array(matches), axis=0).astype(int).tolist()

def get_mcts_player_name(gen: int, n_iters: int):
    return f'MCTS-{gen}-{n_iters}'

def get_model_path(gen: int) -> str:
    return f'/workspace/repo/output/{game}/{tag}/models/gen-{gen}.pt'

def get_mcts_player_str(gen: int, n_iters: int):
    name = get_mcts_player_name(gen, n_iters)

    if gen != 0:
      player_args = {
          '--type': 'MCTS-C',
          '--name': name,
          '-i': n_iters,
          '-m': get_model_path(gen),
          '-n': 1,
      }
    else:
        player_args = {
            '--type': 'MCTS-C',
            '--name': name,
            '-i': n_iters,
            '-n': 1,
            '--no-model': None,
        }

    return make_args_str(player_args)

def run_match_helper(agent1: Agent, agent2: Agent, n_games=100):

    ps1 = get_mcts_player_str(agent1.gen, agent1.n_iters)
    ps2 = get_mcts_player_str(agent2.gen, agent2.n_iters)

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

class BenchmarkCommittee:
    def __init__(self):
      self.W_matrix = np.zeros((0, 0), dtype=float)
      self.G = nx.Graph()
      self.benchmarking_db_conn_pool = DatabaseConnectionPool(database, constants.BENCHMARKING_TABLE_CREATE_CMDS)
      self.ratings = None
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
            agent1 = Agent(gen=gen1, n_iters=gen_iters1)
            agent2 = Agent(gen=gen2, n_iters=gen_iters2)
            if agent1 not in self.G.nodes:
                ix1 = len(self.G.nodes)
                self.G.add_node(agent1, ix=ix1)
                self.expand_matrix()
            else:
                ix1 = self.G.nodes[agent1]['ix']

            if agent2 not in self.G.nodes:
                ix2 = len(self.G.nodes)
                self.G.add_node(agent2, ix=ix2)
                self.expand_matrix()
            else:
                ix2 = self.G.nodes[agent2]['ix']

            if not self.G.has_edge(agent1, agent2):
                self.G.add_edge(agent1, agent2)

            counts = WinLossDrawCounts(gen1_wins, gen2_wins, draws)
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def commit_counts(self, agent1: Agent, agent2: Agent, record: WinLossDrawCounts):
        conn = self.benchmarking_db_conn_pool.get_connection()
        gen1 = agent1.gen
        gen2 = agent2.gen
        n_iters1 = agent1.n_iters
        n_iters2 = agent2.n_iters
        match_tuple = (tag, gen1, gen2, n_iters1, n_iters2, record.win, record.loss, record.draw)
        c = conn.cursor()
        c.execute('INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def play_matches(self, gen: int, n_iters: int):
        matches = get_matches(gen)
        for (x, y) in tqdm(matches):
            agent1 = Agent(gen=x, n_iters=n_iters)
            if agent1 not in self.G.nodes:
                ix1 = len(self.G.nodes)
                self.G.add_node(agent1, ix=ix1)
                self.expand_matrix()
            else:
                ix1 = self.G.nodes[agent1]['ix']

            agent2 = Agent(gen=y, n_iters=n_iters)
            if agent2 not in self.G.nodes:
                ix2 = len(self.G.nodes)
                self.G.add_node(agent2, ix=ix2)
                self.expand_matrix()
            else:
                ix2 = self.G.nodes[agent2]['ix']
            if not self.G.has_edge(agent1, agent2):
                self.G.add_edge(agent1, agent2)
                result = run_match_helper(agent1, agent2)
                self.W_matrix[ix1, ix2] += result.win + 0.5 * result.draw
                self.W_matrix[ix2, ix1] += result.loss + 0.5 * result.draw
                self.commit_counts(agent1, agent2, result)

    def compute_ratings(self):
        ratings = compute_ratings(self.W_matrix)
        self.ratings = {agent: ratings[ix] for agent, ix in self.G.nodes(data='ix')}

if __name__ == '__main__':
    committee = BenchmarkCommittee()
    committee.play_matches(512, 0)
    committee.play_matches(512, 100)
    committee.compute_ratings()

    ratings = committee.ratings
    agent_labels = np.array(list(ratings.keys())).astype(str)
    plt.plot(agent_labels, ratings.values(), label='Rating')
    plt.legend()
    plt.show()
    plt.savefig("ratings.png")

    plt.figure()
    pos = graphviz_layout(committee.G, prog="dot")

    labels = {node: f"{node.gen}" for node in committee.G.nodes}
    nx.draw(committee.G, pos, with_labels=False, node_size=100)
    nx.draw_networkx_labels(committee.G, pos, labels=labels)

    plt.show()
    plt.savefig("my_graph.png")

    plt.figure()
    num_of_matches = []
    num_of_gens = []
    num_of_nodes = []
    for i in range(21):
        m = get_matches(2**i)
        num_of_matches.append(len(m))
        num_of_gens.append(2**i)
        num_of_nodes.append(len(np.unique(m)))
        print(f"{2**i}: {np.unique(m)}")
    plt.plot(num_of_gens, num_of_matches, label='Number of Matches')
    plt.plot(num_of_gens, num_of_nodes, label='Number of Gens')
    plt.legend()
    plt.show()
    plt.savefig("num_matches.png")