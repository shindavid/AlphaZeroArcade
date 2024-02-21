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
from dataclasses import dataclass, fields
from typing import Any, Dict, List


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

    @staticmethod
    def create(s: str, eval_dict: Dict[str, Any]) -> 'WindowSizeFunction':
        eval_dict = dict(eval_dict)
        eval_dict.update(VALID_WINDOW_SIZE_FUNCTIONS)

        obj = None
        try:
            obj = eval(s, eval_dict)
        except Exception as e:
            raise ValueError(f'Failed to evaluate window size function: {s}') from e
        assert isinstance(obj, WindowSizeFunction), f'Invalid window size function: {s}'
        obj.set_repr(s)
        return obj


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


@dataclass
class SamplingParams:
    _window_size_function_str: str = 'katago(n0=minibatches_per_epoch*minibatch_size)'
    target_sample_rate: float = 8
    minibatches_per_epoch: int = 2048
    minibatch_size: int = 256
    window_size_function: WindowSizeFunction = None

    def __post_init__(self):
        attrs = {f.name: getattr(self, f.name) for f in fields(self)}
        self.window_size_function = WindowSizeFunction.create(self._window_size_function_str, attrs)

    def samples_per_window(self):
        return self.minibatch_size * self.minibatches_per_epoch

    @staticmethod
    def create(args) -> 'SamplingParams':
        return SamplingParams(
            _window_size_function_str=args.window_size_function,
            target_sample_rate=args.target_sample_rate,
            minibatches_per_epoch=args.minibatches_per_epoch,
            minibatch_size=args.minibatch_size,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Sampling options')
        defaults = SamplingParams()

        valid_functions_str = str(list(VALID_WINDOW_SIZE_FUNCTIONS.keys()))
        group.add_argument('--window-size-function', default=defaults._window_size_function_str,
                           help=f'window size function (valid functions: {valid_functions_str}, default: "%(default)s")')
        group.add_argument(
            '--target-sample-rate', type=int, default=defaults.target_sample_rate,
            help='target number of times to train over a single row (default: %(default)s))')
        group.add_argument('--minibatches-per-epoch', type=int,
                           default=defaults.minibatches_per_epoch,
                           help='minibatches per epoch (default: %(default)s)')
        group.add_argument('--minibatch-size', type=int, default=defaults.minibatch_size,
                            help='minibatch size (default: %(default)s)')

    def add_to_cmd(self, cmd: List[str]):
        defaults = SamplingParams()
        if self._window_size_function_str != defaults._window_size_function_str:
            cmd.extend(['--window-size-function', self._window_size_function_str])
        if self.target_sample_rate != defaults.target_sample_rate:
            cmd.extend(['--target-sample-rate', str(self.target_sample_rate)])
        if self.minibatches_per_epoch != defaults.minibatches_per_epoch:
            cmd.extend(['--minibatches-per-epoch', str(self.minibatches_per_epoch)])
        if self.minibatch_size != defaults.minibatch_size:
            cmd.extend(['--minibatch-size', str(self.minibatch_size)])


def get_required_dataset_size(params: SamplingParams, prev_window: Window):
    """
    Returns the minimum dataset size that would permit sampling a new window. This corresponds to
    n in the diagram above. The value of c can be computed from n, via:

    c = n - f(n)

    where f = params.window_size_function
    """
    n_samples_per_window = params.samples_per_window()

    a = prev_window.start
    b = prev_window.end
    r = prev_window.sample_rate
    s = n_samples_per_window
    t = params.target_sample_rate
    f = params.window_size_function

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
