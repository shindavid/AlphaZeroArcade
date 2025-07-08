from alphazero.logic.agent_types import Agent, AgentRole, IndexedAgent, MCTSAgent
from alphazero.logic.custom_types import Generation
from alphazero.logic.arena import Arena, RatingData
from alphazero.logic.match_runner import Match, MatchType
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.rating_db import RatingDB
from util.index_set import IndexSet

import numpy as np

import copy
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class RatingsGap:
    left_gen: int
    right_gen: int
    elo_diff: float


@dataclass
class BenchmarkRatingData:
    iagents: List[IndexedAgent]
    ratings: np.ndarray
    committee: IndexSet
    tag: str


class Benchmarker:
    """
    Manages a collection of Agents, their pairwise matches, and rating calculations.
    """
    def __init__(self, organizer: DirectoryOrganizer, db_filename: Optional[str]=None):
        self._organizer = organizer
        self._arena = Arena()

        if db_filename is not None:
            self.db = RatingDB(db_filename)
        else:
            self.db = RatingDB(self._organizer.benchmark_db_filename)
        self.load_from_db()

    def load_from_db(self):
        self._arena.load_agents_from_db(self.db, role=AgentRole.BENCHMARK)
        self._arena.load_matches_from_db(self.db, type=MatchType.BENCHMARK)

    def get_next_matches(self, n_iters, target_elo_gap, n_games,
                         excluded_indices: IndexSet) -> List[Match]:
        """
        The algorithm for selecting the next batch of matches is as follows:
        1. If there are no matches in the arena, play the first generation against the last
        generation. This is the first generation of the benchmark.
        2. If there are matches in the arena, check if there are any incomplete generations.
        3. If there are incomplete generations, let it be the next generation to be played and play
        it against all other generations.
        4. If there are no incomplete generations, check if the biggest gap in ratings is greater
        than the target elo gap. If it is, play the next generation in the middle of the two generations
        with the biggest gap.
        """
        if self.has_no_matches():
            gen0_agent = self.build_agent(0, n_iters)
            last_gen = self._organizer.get_latest_model_generation()
            if last_gen is None or last_gen <2:
                return []
            last_gen_agent = self.build_agent(last_gen, n_iters)
            self._arena.add_agent(gen0_agent, {AgentRole.BENCHMARK}, expand_matrix=True, db=self.db)
            self._arena.add_agent(last_gen_agent, {AgentRole.BENCHMARK}, expand_matrix=True, db=self.db)
            return [Match(gen0_agent, last_gen_agent, n_games, MatchType.BENCHMARK)]

        incomplete_gen = self.incomplete_gen(excluded_indices=excluded_indices)
        if incomplete_gen is not None:
            next_gen = incomplete_gen
            logger.debug('Finishing incomplete gen: %d', next_gen)
        else:
            gap = self.get_biggest_mcts_ratings_gap()
            if gap is None or gap.elo_diff < target_elo_gap:
                return []
            next_gen = (gap.left_gen + gap.right_gen) // 2
            logger.debug('Adding new gen: %d, gap [%d, %d]: %f', next_gen, gap.left_gen, gap.right_gen, gap.elo_diff)

        next_agent = self.build_agent(next_gen, n_iters)
        next_iagent = self._arena.add_agent(next_agent, {AgentRole.BENCHMARK}, expand_matrix=True,
                                             db=self.db)
        matches = self.get_unplayed_matches(next_iagent, n_games, excluded_indices=excluded_indices)
        return matches

    def get_unplayed_matches(self, iagent: IndexedAgent, n_games: int, excluded_indices: IndexSet) -> List[Match]:
        """
        Returns a list of matches that have not been played for the given IndexedAgent, only for the
        agents that are not in the exclude_agents set.
        """
        matches = []
        for ia in self._arena.indexed_agents:
            if ia.agent.gen == iagent.agent.gen:
                continue
            if ia.index in excluded_indices:
                continue
            if self._arena.adjacent_matrix()[iagent.index, ia.index] == False:
                matches.append(Match(iagent.agent, ia.agent, n_games, MatchType.BENCHMARK))
                logger.debug(f'Unplayed match: {iagent.agent.gen} vs {ia.agent.gen}')
        return matches

    def incomplete_gen(self, excluded_indices: IndexSet) -> Optional[Generation]:
        """
        A set of agents is complete if every agent has played every other agent. The set of agents
        is defined as the set of agents in the arena, excluding the agents in the exclude_agents set.
        Returns the generation of the agent that has played the least number of matches.
        Returns None if all agents have played every other agent.
        """

        A = self._arena.adjacent_matrix()
        included_indices = ~excluded_indices.resize(A.shape[0])
        mask = np.where(included_indices)[0]
        A = A[mask][:, mask]
        num_opponents_played = np.sum(A, axis=1)
        incomplete = np.where(num_opponents_played < A.shape[0] - 1)[0]

        if (incomplete.size == 0):
            return None
        incomplete_ix = mask[np.argmin(num_opponents_played)]
        return self._arena.indexed_agents[incomplete_ix].agent.gen

    def get_biggest_mcts_ratings_gap(self) -> Optional[RatingsGap]:
        """
        At any point, the sorted list of mcts-ratings looks like:

        gen_1, rating_1
        gen_2, rating_2
        ...
        gen_n, rating_n

        with rating_1 <= rating_2 <= ... <= rating_n

        Among all i in the range [1...n] with the property that g_i + 1 < G_i, identifies the one
        for which rating_{i_1} - rating_i is maximal, and returns the corresponding gap.

        Here,

        g_i = min(gen_i, gen_{i+1})
        G_i = max(gen_i, gen_{i+1})

        If no such i exists, returns None.

        NOTE: if we want to do partial-gens (i.e., use -i/--num-mcts-iters), then we can change this
        appropriately.
        """
        self.refresh_ratings()
        elos = self._arena.ratings
        gens, agent_ix = zip(*[(indexed_agent.agent.gen, indexed_agent.index) for indexed_agent in self._arena.indexed_agents])
        gens = np.array(gens)
        agent_ix = np.array(agent_ix)

        sorted_ix = np.argsort(gens)
        gaps = np.abs(np.diff(elos[sorted_ix]))
        sorted_gap_ix = np.argsort(gaps)[::-1]
        for gap_ix in sorted_gap_ix:
            left_gen = int(gens[sorted_ix[gap_ix]])
            right_gen = int(gens[sorted_ix[gap_ix + 1]])
            if left_gen + 1 < right_gen:
                logger.debug(f'Found gap: {left_gen} vs {right_gen}, elo diff: {gaps[gap_ix]}')
                return RatingsGap(left_gen, right_gen, float(gaps[gap_ix]))
        logger.debug('No gaps found')
        return None

    def build_agent(self, gen: int, n_iters):
        if gen == 0:
            return MCTSAgent(gen=gen,
                             n_iters=n_iters,
                             tag=self._organizer.tag)
        else:
            return MCTSAgent(gen=gen,
                             n_iters=n_iters,
                             set_temp_zero=True,
                             tag=self._organizer.tag)

    @staticmethod
    def select_committee(elos: np.ndarray, target_elo_gap: float) -> IndexSet:
        """
        Selects a committee of generations that are spaced out by the target elo gap.
        The committee is selected by sorting the generations by their elo ratings, and then
        selecting the highest rated agent, and then selecting the next agent that is
        spaced out by at least the target elo gap. This is done until all agents are scanned.
        The committee is returned as a set of IndexedAgent objects.
        """
        sorted_ix = np.argsort(elos)[::-1]
        sorted_elos = elos[sorted_ix]
        committee_ixs = [sorted_ix[0]]
        last_selected_elo = sorted_elos[0]

        for i in range(1, len(sorted_elos)):
            if last_selected_elo - sorted_elos[i] >= target_elo_gap:
                committee_ixs.append(sorted_ix[i])
                last_selected_elo = sorted_elos[i]

        committee = np.zeros(len(elos), dtype=bool)
        committee[committee_ixs] = True
        return IndexSet.from_bits(committee)

    def has_no_matches(self):
        return self._arena.num_matches() == 0

    def clone_arena(self) -> Arena:
        return copy.deepcopy(self._arena)

    def read_ratings_from_db(self) -> BenchmarkRatingData:
        rating_data: RatingData = self._arena.load_ratings_from_db(self.db, AgentRole.BENCHMARK)
        ratings = rating_data.ratings
        iagents = [self._arena.agent_lookup_db_id[db_id] for db_id in rating_data.agent_ids]
        committee = rating_data.committee
        return BenchmarkRatingData(iagents, ratings, committee, rating_data.tag)

    def refresh_ratings(self):
        self._arena.refresh_ratings()

    @property
    def indexed_agents(self) -> List[IndexedAgent]:
        return self._arena.indexed_agents

    @property
    def agent_lookup(self) -> Dict[Agent, IndexedAgent]:
        return self._arena.agent_lookup

    @property
    def adjacent_matrix(self) -> np.ndarray:
        return self._arena.adjacent_matrix()

    @property
    def ratings(self) -> np.ndarray:
        return self._arena.ratings
