#!/usr/bin/env python3
from alphazero.logic.agent_types import AgentRole, ReferenceAgent
from alphazero.logic.arena import Arena
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.rating_db import RatingDB
from games.game_spec import GameSpec
from games.index import ALL_GAME_SPECS
from util.index_set import IndexSet
from util.logging_util import LoggingParams, configure_logger

import numpy as np

import logging
import os
from typing import List


REF_DIR = os.path.join('/workspace/repo/reference_benchmarks')
os.makedirs(REF_DIR, exist_ok=True)
logger = logging.getLogger(__name__)
configure_logger()


class ReferenceBenchmarker:
    def __init__(self, game_spec: GameSpec):
        assert game_spec.reference_player_family is not None, \
            f'Game {game_spec.name} does not have a reference player family'

        self.game_spec = game_spec
        self.db_filename = os.path.join(REF_DIR, f'{game_spec.name}.db')
        self.db = RatingDB(self.db_filename)
        self.arena = Arena()

    def load_from_db(self):
        self.arena.load_agents_from_db(self.db, role=AgentRole.BENCHMARK)
        self.arena.load_matches_from_db(self.db, type=MatchType.BENCHMARK)

    def add_ref_agents(self):
        strengths = range(self.reference_players.min_strength,
                          self.reference_players.max_strength + 1)
        for strength in strengths:
            agent = ReferenceAgent(type_str=self.reference_players.type_str,
                                   strength_param=self.reference_players.strength_param,
                                   strength=strength)
            self.arena.add_agent(agent, roles=[AgentRole.BENCHMARK], expand_matrix=True, db=self.db)

    def get_matches(self, n_games) -> list[Match]:
        matches = []
        for i in range(1, len(self.arena.indexed_agents)):
            match = Match(agent1=self.arena.indexed_agents[i-1].agent,
                          agent2=self.arena.indexed_agents[i].agent,
                          n_games=n_games,
                          type=MatchType.BENCHMARK)
            matches.append(match)
        return matches

    def play_matches(self, n_games: int=100):
        matches = self.get_matches(n_games)
        binary = os.path.join('/workspace/repo/target/Release/bin', self.game)
        self.arena.play_matches(matches, binary=binary, db=self.db)

    def commit_benchmark_ratings(self):
        self.arena.refresh_ratings()
        committee = IndexSet.from_bits(np.ones(len(self.arena.indexed_agents), dtype=bool))
        self.db.commit_ratings(self.arena.indexed_agents,
                               self.arena.ratings,
                               committee=committee)

    @property
    def reference_players(self):
        return self.game_spec.reference_player_family

    @property
    def game(self):
        return self.game_spec.name


def benchmark_reference_players(game_specs: List[GameSpec]):
    for game_spec in game_specs:
        if game_spec.reference_player_family is None:
            continue
        logger.info(f'Benchmarking reference players for game {game_spec.name}')
        benchmarker = ReferenceBenchmarker(game_spec)
        benchmarker.load_from_db()
        benchmarker.add_ref_agents()
        benchmarker.play_matches()
        benchmarker.commit_benchmark_ratings()
        logger.info(f'Finished. Saved db to {benchmarker.db_filename}')

if __name__ == "__main__":
    benchmark_reference_players(ALL_GAME_SPECS)
