import collections
from typing import Dict

import numpy as np


BETA_SCALE_FACTOR = 100.0 / np.log(1/.36 - 1)  # 100-point difference corresponds to 64% win-rate to match Elo


def int_parse(s: str, prefix: str):
    assert s.startswith(prefix), s
    return int(s[len(prefix):])


class WinLossDrawCounts:
    def __init__(self, win=0, loss=0, draw=0):
        self.win = win
        self.loss = loss
        self.draw = draw

    @property
    def n_games(self):
        return self.win + self.loss + self.draw

    def win_rate(self):
        n = self.n_games
        if n == 0:
            return 0
        return (2 * self.win + self.draw) / (2 * n)

    def to_json(self):
        return [self.win, self.loss, self.draw]

    @staticmethod
    def from_json(l: list):
        return WinLossDrawCounts(*l)

    def __iadd__(self, other):
        self.win += other.win
        self.loss += other.loss
        self.draw += other.draw
        return self

    def __str__(self):
        return f'W{self.win} L{self.loss} D{self.draw}'

    def __repr__(self):
        return str(self)


class MatchRecord:
    def __init__(self):
        self.counts: Dict[int, WinLossDrawCounts] = collections.defaultdict(WinLossDrawCounts)

    def update(self, player_id: int, counts: WinLossDrawCounts):
        self.counts[player_id] += counts

    def get(self, player_id: int) -> WinLossDrawCounts:
        return self.counts[player_id]

    def empty(self) -> bool:
        return len(self.counts) == 0

    def to_json(self):
        return {str(k): wld.to_json() for k, wld in self.counts.items()}

    @staticmethod
    def from_json(d: Dict[str, list]):
        record = MatchRecord()
        for k, v in d.items():
            record.counts[int(k)] = WinLossDrawCounts.from_json(v)
        return record


def extract_match_record(stdout: str) -> MatchRecord:
    """
    Parses the stdout of the c++ binary that runs the matches.

    Looks for this text:

    ...
    All games complete!
    pid=0 name=foo W40 L24 D0 [40]
    pid=1 name=bar W24 L40 D0 [24]
    ...
    """
    record = MatchRecord()
    for line in stdout.splitlines():
        tokens = line.split()
        if len(tokens) > 1 and tokens[0].startswith('pid=') and tokens[0][4:].isdigit():
            player_id = int_parse(tokens[0], 'pid=')
            win = int_parse(tokens[-4], 'W')
            loss = int_parse(tokens[-3], 'L')
            draw = int_parse(tokens[-2], 'D')
            counts = WinLossDrawCounts(win, loss, draw)
            record.update(player_id, counts)

    assert not record.empty(), stdout
    counts1 = record.get(0)
    counts2 = record.get(1)
    assert (counts1.win, counts1.loss, counts1.draw) == (counts2.loss, counts2.win, counts2.draw)
    return record


def compute_ratings(w: np.ndarray) -> np.ndarray:
    """
    Accepts an (n, n)-shaped matrix w, where w[i, j] is the number of wins player i has over player j.

    Outputs a length-n array beta, where beta[i] is the rating of player i.

    Fixes beta[0] = 0 arbitrarily.
    """
    eps = 1e-6
    n = w.shape[0]
    assert w.shape == (n, n)
    assert np.all(w >= 0)
    assert w.diagonal().sum() == 0
    ww = w + w.T
    W = np.sum(w, axis=1)

    p = np.ones(n, dtype=np.float64)
    while True:
        pp = p.reshape((-1, 1)) + p.reshape((1, -1))
        wp_sum = np.sum(ww / pp, axis=1)
        gradient = W / p - wp_sum
        max_gradient = np.max(np.abs(gradient))
        if max_gradient < eps:
            break

        q = W / wp_sum
        q /= q[0]  # so that Random agent's rating is 0
        p = q

    beta = np.log(p) * BETA_SCALE_FACTOR
    return beta
