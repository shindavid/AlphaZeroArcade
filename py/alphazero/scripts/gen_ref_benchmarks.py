#!/usr/bin/env python3
from alphazero.logic.agent_types import AgentRole, ReferenceAgent
from alphazero.logic.arena import Arena
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.rating_db import RatingDB
from games.game_spec import GameSpec
from games.index import ALL_GAME_SPECS

import os


REF_DIR = os.path.join('/workspace/repo/reference_benchmarks')
os.makedirs(REF_DIR, exist_ok=True)


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

    def get_matches(self, n_games: int=1000) -> list[Match]:
        matches = []
        for i in range(1, len(self.arena.indexed_agents)):
            match = Match(agent1=self.arena.indexed_agents[i-1],
                          agent2=self.arena.indexed_agents[i],
                          n_games=n_games,
                          type=MatchType.BENCHMARK)
            matches.append(match)
        return matches

    def play_matches(self, n_games: int=1000):
        matches = self.get_matches(n_games)
        self.arena.play_matches(matches, game=self.game_spec.name, db=self.db)


    def refresh_ratings(self):
        self.arena.refresh_ratings()

    @property
    def reference_players(self):
        return self.game_spec.reference_player_family

    @property
    def game(self):
        return self.game_spec.name


def benchmark_reference_players(game_spec: GameSpec):
    pass

if __name__ == "__main__":
    benchmark_reference_players(ALL_GAME_SPECS[1])
