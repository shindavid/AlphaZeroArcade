from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.arena import Arena, RatingArrays
from alphazero.logic.match_runner import Match
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.rating_db import RatingDB
from util.logging_util import get_logger

from dataclasses import dataclass
import numpy as np
from typing import  Optional, List


logger = get_logger()


@dataclass
class RatingsGap:
    left_gen: int
    right_gen: int
    elo_diff: float


class Benchmarker:
    """
    Manages a collection of Agents, their pairwise matches, and rating calculations.
    """
    def __init__(self, organizer: DirectoryOrganizer):
        self._organizer = organizer
        self._arena = Arena()
        self._ratings: np.ndarray = np.array([])
        self._committee_ixs: np.ndarray = np.array([])
        self._db = RatingDB(self._organizer.benchmark_db_filename)
        self.load_from_db()

    def load_from_db(self):
        self._arena.load_agents_from_db(self._db)
        self._arena.load_matches_from_db(self._db)
        rating_arrays: RatingArrays = self._arena.load_ratings_from_db(self._db)
        self._ratings = rating_arrays.ratings
        self._committee_ixs = rating_arrays.committee_ixs
        if self._ratings.size > 0:
            assert len(self._ratings) == len(self.indexed_agents)
        if self._ratings.size == 0 and not self.has_no_matches():
            self.compute_ratings()

    def run(self, n_iters: int=100, n_games: int=100, target_elo_gap: int=100):
        """
        Runs the benchmarker. The general idea is to find the largest gap in ratings that is greater
        than the target elo gap, and play the next generation in the middle of the two generations
        with the biggest gap until there is no elo gap greater than the target elo gap. The elo ratings
        are recomputed after each batch of matches (for a generation to be played against). After this,
        we have a set of generations whose elo rating gaps are all within the target elo gap. To avoid
        having two generations with ratings that are too similar, we select a committee of generations
        that are spaced out by the target elo gap.
        """
        while True:
            matches = self.get_next_matches(n_iters, target_elo_gap, n_games)
            if matches is None:
                break
            self._arena.play_matches(matches, self._organizer.game, additional=False, db=self._db)
            self.compute_ratings()

        is_committee = self.select_committee(target_elo_gap)
        self.compute_ratings()
        self._arena.commit_ratings_to_db(self._db, self.indexed_agents, self._ratings, is_committee)

    def get_next_matches(self, n_iters, target_elo_gap, n_games):
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
            last_gen_agent = self.build_agent(last_gen, n_iters)
            return [Match(gen0_agent, last_gen_agent, n_games)]

        incomplete_gen = self.incomplete_gen()
        if incomplete_gen is not None:
            next_gen = incomplete_gen
            logger.info('Finishing incomplete gen: %d', next_gen)
        else:
            gap = self.get_biggest_mcts_ratings_gap()
            if gap is None or gap.elo_diff < target_elo_gap:
                return None
            next_gen = (gap.left_gen + gap.right_gen) // 2
            logger.info('Adding new gen: %d, gap [%d, %d]: %f', next_gen, gap.left_gen, gap.right_gen, gap.elo_diff)

        next_agent = self.build_agent(next_gen, n_iters)
        matches = [Match(next_agent, indexed_agent.agent, n_games) for indexed_agent in self.indexed_agents]
        return matches

    def incomplete_gen(self) -> np.ndarray:
        """
        If a run was interrupted, there may be generations that have not been played against all
        other generations. This function identifies the newly added generation that was in the middle
        of being played against all other generations, and returns it to be the next gen to be played.
        """
        A = self._arena.adjacent_matrix()
        num_opponents_played = np.sum(A, axis=1)
        incomplete = np.where(num_opponents_played < len(self.indexed_agents) - 1)[0]
        if (incomplete.size == 0):
            return None
        incomplete_ix = np.argmin(num_opponents_played)
        return self.indexed_agents[incomplete_ix].agent.gen

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
        elos = self._ratings.copy()
        gens, agent_ix = zip(*[(indexed_agent.agent.gen, indexed_agent.index) for indexed_agent in self.indexed_agents])
        gens = np.array(gens)
        agent_ix = np.array(agent_ix)

        sorted_ix = np.argsort(gens)
        gaps = np.abs(np.diff(elos[sorted_ix]))
        sorted_gap_ix = np.argsort(gaps)[::-1]
        for gap_ix in sorted_gap_ix:
            left_gen = int(gens[sorted_ix[gap_ix]])
            right_gen = int(gens[sorted_ix[gap_ix + 1]])
            if left_gen + 1 < right_gen:
                return RatingsGap(left_gen, right_gen, float(gaps[gap_ix]))

    def compute_ratings(self, eps=1e-3):
        self._ratings = self._arena.compute_ratings(eps=eps)

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

    def select_committee(self, target_elo_gap) -> List[bool]:
        """
        Selects a committee of generations that are spaced out by the target elo gap.
        The committee is selected by first finding the min and max elo ratings, and then
        dividing the range into equal intervals of size target_elo_gap. The generations
        that are closest to the target elos are selected. The committee is then returned
        as a boolean mask, where True indicates that the generation is in the committee.
        """
        elos = self._ratings.copy()
        max_elo = np.max(elos)
        min_elo = np.min(elos)
        committee_size = int((max_elo - min_elo) / target_elo_gap)
        target_elos = np.linspace(min_elo, max_elo, committee_size)

        sorted_ix = np.argsort(elos)
        committee_ixs = []
        j = 0
        n = len(elos)
        for e in target_elos:
            while j < n - 1 and (abs(elos[sorted_ix[j]] - e) > abs(elos[sorted_ix[j + 1]] - e)):
                j += 1
            committee_ixs.append(sorted_ix[j])
        committee_ixs = np.unique(committee_ixs)
        self._committee_ixs = committee_ixs

        mask = np.zeros(len(elos), dtype=bool)
        mask[committee_ixs] = True
        return mask.astype(int).tolist()

    def has_no_matches(self):
        return self._arena.num_matches() == 0

    def clone_arena(self) -> Arena:
        """
        Note: the agents in the cloned arena are the same instances as the original arena.
        """
        return self._arena.clone()

    @property
    def indexed_agents(self):
        return self._arena.indexed_agents

    @property
    def ratings(self):
        return self._ratings.copy()

    @property
    def committee_ixs(self):
        return self._committee_ixs.copy()