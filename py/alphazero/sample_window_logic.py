"""
This module contains logic for maintaining the sampling window.

There is an ever-growing master-list M of all positions. Within M, there is a most-recently
selected sample window, W = M[a:b]. The positions of W have been sampled an average of r times each.

The next window to use will be W' = M[c:] for some c.

Here is a diagram of the above:

                                    [---------W'---------]
                              [--------W---------]
M: ---------------------------+-----+------------+-------+
                              a     c            b       n

Given W (represented by {a, b, r}), at what point can we start sampling the next window (W')?

This module provides functionality for answering this question.
"""

import abc
import argparse
from dataclasses import dataclass
from typing import List


class WindowSizeFunction(abc.ABC):
    def __init__(self):
        self._repr = '???'

    def set_repr(self, s: str):
        self._repr = s

    def __repr__(self):
        return self._repr

    def __str__(self):
        return repr(self)

    @abc.abstractmethod
    def __call__(self, n: int) -> int:
        """
        Returns the size of the next window to use, given that the master list M has size n.
        """
        pass


class FixedWindowSizeFunction(WindowSizeFunction):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    def __call__(self, n: int) -> int:
        return self.window_size


class KataGoWindowSizeFunction(WindowSizeFunction):
    """
    Uses the function

    f(n) = n^alpha

    rescaled so that f(n0) = n0 and so that f'(n0) = beta.

    See Appendix C of the KataGo paper for details. KataGo uses n0=250,000, but I think this
    parameter should in general be whatever size your generation-0 dataset is.
    """
    def __init__(self, n0, alpha=0.75, beta=0.4):
        super().__init__()
        self.n0 = n0
        self.alpha = alpha
        self.beta = beta

    def __call__(self, n: int) -> int:
        gamma = (n / self.n0)**self.alpha - 1
        return min(n, self.n0 * (1 + self.beta * gamma / self.alpha))


@dataclass
class Window:
    """
    In the notation of the diagram above, this class represents a window W = M[a:b] together with a
    sample rate r.

    Assuming a constant r for the entire window is a simplification that makes the math easier. This
    will tend to overestimate the end of the window and underestimate the start of the window.

    This simplification is reasonable because as the sample window slides forward, each position
    slides from the end of the window to the start of the window, and so the overestimates and
    the underestimates will approximately cancel out.
    """
    start: int
    end: int
    sample_rate: float


VALID_WINDOW_SIZE_FUNCTIONS = {
    'katago': KataGoWindowSizeFunction,
    'fixed': FixedWindowSizeFunction,
}


class SamplingParams:
    window_size_function: WindowSizeFunction
    target_sample_rate: float
    minibatches_per_epoch: int
    minibatch_size: int

    @staticmethod
    def samples_per_window():
        return SamplingParams.minibatch_size * SamplingParams.minibatches_per_epoch

    @staticmethod
    def load(args):
        SamplingParams.target_sample_rate = args.target_sample_rate
        SamplingParams.minibatches_per_epoch = args.minibatches_per_epoch
        SamplingParams.minibatch_size = args.minibatch_size

        static_attrs = {k: getattr(SamplingParams, k) for k in dir(SamplingParams)}
        static_attrs = {k: v for k, v in static_attrs.items() if
                        not k.startswith('_') and not callable(v)}

        eval_dict = dict(VALID_WINDOW_SIZE_FUNCTIONS)
        eval_dict.update(static_attrs)

        SamplingParams.window_size_function = eval(args.window_size_function, eval_dict)
        SamplingParams.window_size_function.set_repr(args.window_size_function)
        assert isinstance(SamplingParams.window_size_function, WindowSizeFunction)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Sampling options')

        valid_functions_str = str(list(VALID_WINDOW_SIZE_FUNCTIONS.keys()))
        default_fn = 'katago(n0=minibatches_per_epoch*minibatch_size)'
        group.add_argument('--window-size-function', default=default_fn,
                           help=f'window size function (valid functions: {valid_functions_str}, default: %(default)s)')
        group.add_argument(
            '--target-sample-rate', type=int, default=8,
            help='target number of times to train over a single row (default: %(default)s))')
        group.add_argument('--minibatches-per-epoch', type=int, default=2048,
                            help='minibatches per epoch (default: %(default)s)')
        group.add_argument('--minibatch-size', type=int, default=256,
                            help='minibatch size (default: %(default)s)')

    @staticmethod
    def add_to_cmd(cmd: List[str]):
        cmd.extend([
            '--window-size-function', str(SamplingParams.window_size_function),
            '--target-sample-rate', str(SamplingParams.target_sample_rate),
            '--minibatches-per-epoch', str(SamplingParams.minibatches_per_epoch),
            '--minibatch-size', str(SamplingParams.minibatch_size),
            ])


def get_required_dataset_size(prev_window: Window):
    """
    Returns the minimum dataset size that would permit sampling a new window. This corresponds to
    n in the diagram above. The value of c can be computed from n, via:

    c = n - f(n)

    where f = SamplingParams.window_size_function
    """
    n_samples_per_window = SamplingParams.samples_per_window()

    a = prev_window.start
    b = prev_window.end
    r = prev_window.sample_rate
    s = n_samples_per_window
    t = SamplingParams.target_sample_rate
    f = SamplingParams.window_size_function

    """
    At the proper (c, n), we should have:

    (b-c)*r + s <= t*f(n)

    The (b-c)*r term represents samples from the previous window, the s term represents additional
    samples from the next window, and the t*f(n) term represents the target total number of
    samples for the next window.

    Plugging in c = n - f(n), and rearranging, we have:

    b*r + s <= n*r + (t-r)*f(n)

    We binary-search on n to find the smallest value that satisfies this inequality.

    For special cases like FixedSamplingMethod, we can of course solve this more directly. But
    this general method works for any SamplingMethod. This function is not called often so it's ok
    to be a bit slow.
    """
    lhs = b * r + s
    rhs = lambda n: n * r + (t - r) * f(n)

    lo = b

    # print(f'DEBUG a={a} b={b} r={r} s={s} t={t} lhs={lhs} rhs(lo)={rhs(lo)}')
    if lhs <= rhs(lo):
        return b

    hi = b + 2*(b-a)  # 2 prev_window lengths should usually be enough
    inf_loop_protection = 100
    while lhs > rhs(hi) and inf_loop_protection:
        inf_loop_protection -= 1
        hi *= 2

    assert inf_loop_protection, 'Infinite loop detected initializing hi'

    # Binary search between lo and hi to find n
    inf_loop_protection = 100
    while lo < hi and inf_loop_protection:
        inf_loop_protection -= 1
        mid = (lo + hi) // 2
        if lhs > rhs(mid):
            lo = mid + 1
        else:
            hi = mid

    assert inf_loop_protection, 'Infinite loop detected during binary search'

    return lo


def construct_window(prev_window: Window, c: int, n: int, n_sampled_positions: int) -> Window:
    """
    Constructs a new window from a previous window, by incorporating the fact that we just sampled
    n_sampled_positions from M[c:n].
    """
    assert n > c
    n_prev_window_samples = max(0, prev_window.end - c) * prev_window.sample_rate
    n_total_samples = n_prev_window_samples + n_sampled_positions
    sample_rate = n_total_samples / (n - c)
    return Window(c, n, sample_rate)
