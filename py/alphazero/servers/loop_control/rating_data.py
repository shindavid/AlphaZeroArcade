from alphazero.logic.custom_types import ClientId, Generation
from alphazero.logic.ratings import WinLossDrawCounts
from util.logging_util import get_logger

from collections import defaultdict
import math
from typing import Dict, Optional


logger = get_logger()


RatingTag = str


"""
For now, hard-coding this constant.

If we want to make the configurable in the future, we have to decide whether it will be configurable
from the loop-controller side or from the ratings-server side.
"""
N_GAMES = 100


class RatingData:
    """
    A RatingData summarizes match results for a given MCTS generation.
    """

    def __init__(self, mcts_gen: int, min_ref_strength: int, max_ref_strength: int):
        self.mcts_gen = mcts_gen
        self.min_ref_strength = min_ref_strength
        self.max_ref_strength = max_ref_strength

        self.est_rating = None
        self.rating_lower_bound = min_ref_strength - 1
        self.rating_upper_bound = max_ref_strength + 1

        self.owner: Optional[ClientId] = None  # client that is currently evaluating this gen

        self.match_data = defaultdict(WinLossDrawCounts)  # ref_strength -> mcts WLD
        self.rating = None

    def __str__(self):
        return (f'RatingData(mcts_gen={self.mcts_gen}, rating={self.rating}, '
                f'owner={self.owner}, est_rating={self.est_rating}, '
                f'rating_bounds=({self.rating_lower_bound}, {self.rating_upper_bound}), '
                f'match_data={dict(self.match_data)})')

    def __repr__(self):
        return str(self)

    def filtered_match_data(self) -> Dict[int, WinLossDrawCounts]:
        """
        Returns match data for ref strengths that have been tested at least N_GAMES times.

        With the current implementation, this should typically just return a copy of
        self.match_data. The only case where it could be different is if N_GAMES is changed
        between runs.
        """
        return {k: v for k, v in self.match_data.items() if v.n_games >= N_GAMES}

    def add_result(self, ref_strength: int, counts: WinLossDrawCounts, set_rating=True):
        self.match_data[ref_strength] += counts
        counts = self.match_data[ref_strength]

        if counts.n_games < N_GAMES:
            return

        if counts.win_rate() < 0.5:
            self.rating_upper_bound = min(self.rating_upper_bound, ref_strength)
        else:
            self.rating_lower_bound = max(self.rating_lower_bound, ref_strength)

        if set_rating:
            self.set_rating()

    def set_rating(self):
        if self.rating is not None:
            return

        if self.rating_upper_bound == self.min_ref_strength:
            self.rating = self.min_ref_strength
            return

        if self.rating_lower_bound == self.max_ref_strength:
            self.rating = self.max_ref_strength
            return

        if self.rating_lower_bound + 1 < self.rating_upper_bound:
            return

        assert self.rating_lower_bound + 1 == self.rating_upper_bound

        self.rating = self._interpolate_bounds(self.filtered_match_data())

    def _interpolate_bounds(self, match_data):
        """
        Interpolates between the lower and upper bound to estimate the critical strength.

        The interpolation formula is:

        rating = midpoint + spread_factor * adjustment

        where:

        midpoint = 0.5 * (x1 + x2)
        spread_factor = sqrt(x2 - x1)
        adjustment = 0.5 * (w1 - w2) / (w1 + w2)

        x1 = rating_lower_bound
        x2 = rating_upper_bound
        w1 = (win rate at x1) - 0.5
        w2 = 0.5 - (win rate at x2)

        If lower + 1 == upper, then this estimate is exactly the critical strength, and so serves
        as the exact rating.
        """
        lower_counts = match_data.get(self.rating_lower_bound, WinLossDrawCounts(win=1))
        upper_counts = match_data.get(self.rating_upper_bound, WinLossDrawCounts(loss=1))

        x1 = self.rating_lower_bound
        x2 = self.rating_upper_bound
        w1 = lower_counts.win_rate() - 0.5
        w2 = 0.5 - upper_counts.win_rate()

        assert x2 >= x1 + 1, (x1, x2, match_data)
        assert w1 >= 0, (w1, x1, match_data)
        assert w2 > 0, (w2, x2, match_data)

        midpoint = 0.5 * (x1 + x2)
        spread_factor = math.sqrt(x2 - x1)
        adjustment = 0.5 * (w1 - w2) / (w1 + w2)
        strength = midpoint + spread_factor * adjustment

        logger.debug(f'Interpolating bounds for gen={self.mcts_gen} match_data={match_data}: '
                     f'({x1}: {w1}, {x2}: {w2}) -> {midpoint} + {spread_factor} * {adjustment} = {strength}')

        assert x1 <= strength <= x2, (x1, strength, x2, match_data)
        return strength

    def _get_candidates(self, est_rating: float, match_data: Dict[int, WinLossDrawCounts]):
        """
        Returns a list of candidate strengths to test, from best to worst, measured by distance
        to est_rating.

        Valid candidates must satisfy:

        1. abs(candidate - est_rating) < 1
        2. self.rating_lower_bound <= candidate <= self.rating_upper_bound
        3. candidate not in match_data
        """
        left = int(math.floor(est_rating))
        right = int(math.ceil(est_rating))
        candidates = list(set([left, right]))
        candidates = [c for c in candidates if c not in match_data]
        candidates = [c for c in candidates if self.rating_lower_bound <=
                      c <= self.rating_upper_bound]
        candidates.sort(key=lambda c: abs(est_rating - c))
        return candidates

    def get_next_strength_to_test(self):
        if self.rating is not None:
            return None

        logger.debug(f'Getting next strength to test for gen={self.mcts_gen}: '
                     f'est={self.est_rating}, '
                     f'bounds=({self.rating_lower_bound}, {self.rating_upper_bound})')

        match_data = self.filtered_match_data()

        if self.est_rating is not None:
            candidates = self._get_candidates(self.est_rating, match_data)
            for c in candidates:
                return c

        est_rating = self._interpolate_bounds(match_data)
        candidates = self._get_candidates(est_rating, match_data)
        for c in candidates:
            return c
        raise Exception(f'Unexpected state: {self}')


RatingDataDict = Dict[Generation, RatingData]
