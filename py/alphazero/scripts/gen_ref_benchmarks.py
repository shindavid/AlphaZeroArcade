#!/usr/bin/env python3
from alphazero.logic.agent_types import AgentRole, ReferenceAgent
from alphazero.logic.arena import Arena
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.rating_db import RatingDB
from games.game_spec import GameSpec
import games.index as game_index
from games.index import ALL_GAME_SPECS
from util.index_set import IndexSet
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter

import numpy as np

import argparse
import logging
import os
import shlex
import sys
from typing import List


REF_DIR = os.path.join('/workspace/repo/reference_benchmarks')
logger = logging.getLogger(__name__)


class ReferenceBenchmarker:
    def __init__(self, args, game_spec: GameSpec):
        assert game_spec.reference_player_family is not None, \
            f'Game {game_spec.name} does not have a reference player family'

        self.neighborhood_size = args.neighborhood_size
        self.min_elo_gap = args.min_elo_gap
        self.game_spec = game_spec
        self.db_filename = os.path.join('/workspace/output', self.game_spec.name, 'reference.players/databases', 'benchmark.db')
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
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
            roles = {AgentRole.BENCHMARK}
            self.arena.add_agent(agent, roles=roles, expand_matrix=True, db=self.db)

    def get_matches(self, n_games) -> list[Match]:
        matches = []
        A = self.neighborhood_size
        B = len(self.arena.indexed_agents)

        for i in range(B):
            for j in range(max(0, i - A), i):
                match = Match(agent1=self.arena.indexed_agents[i].agent,
                              agent2=self.arena.indexed_agents[j].agent,
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
        committee = Benchmarker.select_committee(self.arena.ratings, self.min_elo_gap)
        self.db.commit_ratings(self.arena.indexed_agents,
                               self.arena.ratings,
                               committee=committee)
        cmd = shlex.join(sys.argv)
        committee_iagents = []
        committee_ratings = []
        for ix in committee:
            committee_iagents.append(self.arena.indexed_agents[ix])
            committee_ratings.append(self.arena.ratings[ix])

        self.db.save_ratings_to_json(committee_iagents, committee_ratings,
                                     os.path.join(REF_DIR, f'{self.game}.json'), cmd)

    def run(self):
        self.load_from_db()
        self.add_ref_agents()
        self.play_matches()
        self.commit_benchmark_ratings()

    @property
    def reference_players(self):
        return self.game_spec.reference_player_family

    @property
    def game(self):
        return self.game_spec.name


def benchmark_reference_players(args, game_specs: List[GameSpec]):
    os.makedirs(REF_DIR, exist_ok=True)
    for game_spec in game_specs:
        if game_spec.reference_player_family is None:
            logger.info(f'Skipped for game: {game_spec.name}, no reference_player_family')
            continue
        logger.info(f'Benchmarking reference players for game {game_spec.name}')
        benchmarker = ReferenceBenchmarker(args, game_spec)
        benchmarker.run()
        logger.info(f'Finished for game: {game_spec.name}')


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    parser.add_argument('-n', '--neighborhood-size', type=int, default=1,
                        help='Neighborhood size (default: %(default)s)')
    game_index.add_parser_argument(parser, '-g', '--game',
                                   help='Comma-separate games. If not specified, all games will be benchmarked.')
    parser.add_argument('--min-elo-gap', type=float, default=200.0, help='Minimum elo gap between generations in the committee (default: %(default)s)')
    LoggingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    logging_params = LoggingParams.create(args)
    configure_logger(params=logging_params)

    if args.game is None:
        specs = ALL_GAME_SPECS
    else:
        specs = [game_index.get_game_spec(g) for g in args.game.split(',')]
    benchmark_reference_players(args, specs)


if __name__ == "__main__":
    main()
