import collections
from typing import Dict, List, Union

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


def extract_match_record(stdout: Union[List[str], str]) -> MatchRecord:
    """
    Parses the stdout of the c++ binary that runs the matches.

    Looks for this text:

    ...
    2024-03-13 14:10:09.131334 All games complete!
    2024-03-13 14:10:09.131368 pid=0 name=Perfect-21 W51 L49 D0 [51]
    2024-03-13 14:10:09.131382 pid=1 name=MCTS-C-100 W49 L51 D0 [49]
    ...

    OR:

    ...
    All games complete!
    pid=0 name=Perfect-21 W51 L49 D0 [51]
    pid=1 name=MCTS-C-100 W49 L51 D0 [49]
    ...
    """
    record = MatchRecord()
    lines = stdout.splitlines() if isinstance(stdout, str) else stdout
    for line in lines:
        tokens = line.split()
        skip_list = [0, 2]  # hacky way to test for the two possible formats
        for skip in skip_list:
            if len(tokens) > skip and tokens[skip].startswith('pid=') and tokens[skip][4:].isdigit():
                player_id = int_parse(tokens[skip], 'pid=')
                win = int_parse(tokens[skip + 2], 'W')
                loss = int_parse(tokens[skip + 3], 'L')
                draw = int_parse(tokens[skip + 4], 'D')
                counts = WinLossDrawCounts(win, loss, draw)
                record.update(player_id, counts)
                break

    assert not record.empty(), stdout
    counts1 = record.get(0)
    counts2 = record.get(1)
    assert (counts1.win, counts1.loss, counts1.draw) == (counts2.loss, counts2.win, counts2.draw), '\n'.join(lines)
    return record


def compute_ratings(w: np.ndarray, eps: float=0.0) -> np.ndarray:
    """
    Accepts an (n, n)-shaped matrix w, where w[i, j] is the number of wins player i has over player j.

    Outputs a length-n array beta, where beta[i] is the rating of player i.

    Fixes beta[0] = 0 arbitrarily.

    Adds eps to each off-diagonal entry of w before performing the calculation. Using a nonzero
    value for eps ensures that ratings will be well-defined, even if w is disconnected.

    TODO: think about interplay between eps and gradient_threshold.
    """
    gradient_threshold = 1e-6

    w = w.copy()
    w = w + eps
    np.fill_diagonal(w, 0)
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
        if max_gradient < gradient_threshold:
            break

        q = W / wp_sum
        q /= q[0]  # so that Random agent's rating is 0
        p = q

    beta = np.log(p) * BETA_SCALE_FACTOR
    return beta

def win_prob(elo1: float, elo2: float) -> float:
    return 1 / (1 + np.exp((elo2 - elo1) / BETA_SCALE_FACTOR))

def estimate_elo_newton(n: np.ndarray, k: np.ndarray, elos: np.ndarray, init: float = 0.0,
                        lower: float = -np.inf, upper: float = np.inf, max_step: float = 200,
                        tol: float = 1e-8, max_iter: int = 100, eps: float = 1e-8) -> float:
    """
    Maximum–likelihood estimate of player T's Elo via Newton's method.

    Parameters
    ----------
    n     : array of total games versus each opponent
    k     : array of T's wins versus each opponent
    elos  : array of opponent Elo ratings (same length as n, k)
    init  : initial guess for R_T
    lower : lower bound on R_T
    upper : upper bound on R_T
    max_step : maximum step size in each Newton iteration
    tol   : convergence threshold on the gradient
    max_iter : cap on Newton iterations
    eps   : small value to ensure Hessian is not too close to zero

    Returns
    -------
    float : MLE estimate of R_T
    """
    R_T = float(init)

    for _ in range(max_iter):
        g = win_prob(R_T, elos)
        grad = (1.0 / BETA_SCALE_FACTOR) * np.sum(k - n * g)
        if abs(grad) < tol:
            break
        hess = -(1.0 / BETA_SCALE_FACTOR**2) * np.sum(n * g * (1.0 - g))
        hess = -max(abs(hess), eps)
        step = grad / hess
        step = np.clip(step, -max_step, max_step)
        R_T -= step

    print(f"Converged after {_+1} iterations with R_T = {R_T}, grad = {grad}, hess = {hess}")

    return np.clip(R_T, lower, upper)

if __name__ == "__main__":
    elos = np.array([795, 1196, 1701], dtype=float)
    n    = np.array([7, 2, 2])
    k    = np.array([2, 0, 0])

    R_hat = estimate_elo_newton(n, k, elos, init=0.0)
    print(f"MLE Elo for T ≈ {R_hat:.2f}")