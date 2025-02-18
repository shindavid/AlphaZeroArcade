from alphazero.logic.ratings import WinLossDrawCounts, compute_ratings, BETA_SCALE_FACTOR
from alphazero.logic.agent_types import Agent, MCTSAgent, PerfectAgent, RandomAgent
from alphazero.logic.match_runner import Match, MatchRunner
from alphazero.logic.rating_db import RatingDB
from alphazero.logic import constants
from util.str_util import make_args_str

from util.logging_util import get_logger


import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm

from typing import List, Dict, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import os
import random

logger = get_logger()

class DirectoryOrganizer:
    def __init__(self, game, tag, db_name=None):
        self.game = game
        self.tag = tag
        if db_name is None:
            self.db_name = 'benchmark'
        else:
            self.db_name = db_name

        self.base_dir = os.path.join('/workspace/repo')
        self.output_dir = os.path.join(self.base_dir, 'output', game, tag)
        self.db_dir = os.path.join(self.output_dir, 'databases')
        self.binary = os.path.join(self.base_dir, 'target/Release/bin', game)
        self.model_dir = os.path.join(self.output_dir, 'models')

class BenchmarkCommittee:
    def __init__(self, organzier: DirectoryOrganizer, load_past_data: bool=False):
        self._organizer = organzier

        self.W_matrix = np.zeros((0, 0), dtype=float)
        self.G = nx.Graph()

        self.rating_db = RatingDB(self._organizer.db_dir, self._organizer.db_name)
        self.binary = self._organizer.binary
        self.ratings = None
        if load_past_data:
            self.load_past_data()

    def load_past_data(self):
        rows = self.rating_db.fetchall()
        for gen1, gen2, gen_iters1, gen_iters2, gen1_wins, gen2_wins, draws in rows:
            agent1 = RatingDB.build_agent_from_row(gen1, gen_iters1, model_dir=self._organizer.model_dir)
            agent2 = RatingDB.build_agent_from_row(gen2, gen_iters2, model_dir=self._organizer.model_dir)
            num_games = gen1_wins + gen2_wins + draws
            ix1, _ = self._add_agent_node(agent1)
            ix2, _ = self._add_agent_node(agent2)

            if self.G.has_edge(agent1, agent2):
                self.G.edges[(agent1, agent2)]['n_games'] += num_games
            else:
                self.G.add_edge(agent1, agent2, n_games=num_games)

            counts = WinLossDrawCounts(gen1_wins, gen2_wins, draws)
            self.W_matrix[ix1, ix2] += counts.win + 0.5 * counts.draw
            self.W_matrix[ix2, ix1] += counts.loss + 0.5 * counts.draw

    def play_matches(self, matches: List[Match], additional=False):
      for match in tqdm(matches):
        ix1, _ = self._add_agent_node(match.agent1)
        ix2, _ = self._add_agent_node(match.agent2)

        if self.G.has_edge(match.agent1, match.agent2):
          if not additional:
            n_games_played = self.G.edges[(match.agent1, match.agent2)]['n_games']
            match.n_games = match.n_games - n_games_played
          if match.n_games < 1:
            continue
        else:
          self.G.add_edge(match.agent1, match.agent2, n_games=0)

        result = MatchRunner.run_match_helper(match, self.binary)
        self.W_matrix[ix1, ix2] += result.win + 0.5 * result.draw
        self.W_matrix[ix2, ix1] += result.loss + 0.5 * result.draw
        self.G[match.agent1][match.agent2]['n_games'] += match.n_games
        self.rating_db.commit_counts(match.agent1, match.agent2, result)

    def compute_ratings(self):
        assert nx.is_connected(self.G)
        ratings = compute_ratings(self.W_matrix).tolist()
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
                ix, is_new_node = sub_committee._add_agent_node(node)
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

    def _add_agent_node(self, agent: Agent) -> int:
        if agent not in self.G.nodes:
            ix = len(self.G.nodes)
            self.G.add_node(agent, ix=ix)
            self._expand_matrix()
            is_new_node = True
        else:
            ix = self.G.nodes[agent]['ix']
            is_new_node = False
        return ix, is_new_node

    def _expand_matrix(self):
        n = self.W_matrix.shape[0]
        new_matrix = np.zeros((n + 1, n + 1), dtype=float)
        new_matrix[:n, :n] = self.W_matrix
        self.W_matrix = new_matrix

class Evaluation:
    def __init__(self, organizer: DirectoryOrganizer, benchmark_committee: BenchmarkCommittee):
        self._organizer = organizer
        self.benchmark_committee = benchmark_committee
        assert self.benchmark_committee.ratings is not None
        self.rating_db = RatingDB(self._organizer.db_dir, self._organizer.db_name)
        self.eval = self.benchmark_committee.sub_committee(organizer=self._organizer)
        self.benchmark_ratings = dict(sorted(self.benchmark_committee.ratings.items(), key=lambda x: x[1]))

    def interpolate_ratings(self, test_agent: Agent, test_group_elo_ratings: Dict[Agent, float]):
        interp_table = {v: self.benchmark_committee.ratings[k] for k, v in test_group_elo_ratings.items() if k != test_agent}
        interp_table = sorted(interp_table.items(), key=lambda x: x[1])
        interp_table = list(zip(*interp_table))
        x_values = interp_table[0]
        y_values = interp_table[1]
        x = test_group_elo_ratings[test_agent]
        test_rating = np.interp(x, x_values, y_values)
        return test_rating

    def evaluate(self, test_agent: Agent, n_games: int=4, n_steps: int=10):
        representatives = []
        init_benchmark_agent = random.choice(list(self.benchmark_committee.G.nodes))
        init_match = Match(test_agent, init_benchmark_agent, n_games)
        self.eval.play_matches([init_match], additional=True)
        representatives.append(init_benchmark_agent)
        for _ in range(n_steps):
            eval_sub_committee = self.eval.sub_committee(include_agents=[test_agent] + list(self.benchmark_committee.G.nodes))
            eval_sub_committee.compute_ratings()
            eval_ratings = eval_sub_committee.ratings
            p = {agent: 1/(1 + np.exp((rating - eval_ratings[test_agent])/BETA_SCALE_FACTOR)) \
                for agent, rating in eval_ratings.items() if agent != test_agent}
            weights = {agent: p * (1 - p) for agent, p in p.items()}
            next_agent = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            next_match = Match(test_agent, next_agent, n_games)
            self.eval.play_matches([next_match], additional=True)
            representatives.append(next_agent)

        test_group_elo_ratings = eval_sub_committee.ratings
        test_rating = self.interpolate_ratings(test_agent, test_group_elo_ratings)
        self.eval.rating_db.commit_rating(test_agent, test_rating, representatives, self._organizer.tag)
        return test_rating

if __name__ == '__main__':
    game = 'c4'
    tag = 'benchmark'
    n_games = 100
    organzier = DirectoryOrganizer(game, tag, db_name='benchmark')
    benchmark_committee = BenchmarkCommittee(organzier, load_past_data=True)
    matches = MatchRunner.linspace_matches(0, 128, n_iters=100, freq=4, n_games=n_games, \
        model_dir=organzier.model_dir)
    benchmark_committee.play_matches(matches)
    benchmark_committee.compute_ratings()

    eval_organizer = DirectoryOrganizer(game, tag, db_name='test_eval')
    evaluation = Evaluation(eval_organizer, benchmark_committee)
    test_agents = [MCTSAgent(gen=128, n_iters=1, model_dir=organzier.model_dir)]
    for test_agent in tqdm(test_agents):
        test_rating = evaluation.evaluate(test_agent, n_games=10, n_steps=10)
        print(f'{test_agent}: {test_rating}')
