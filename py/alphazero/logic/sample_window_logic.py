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
from alphazero.logic.learning_params import LearningParams

from dataclasses import dataclass


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


def get_required_dataset_size(params: LearningParams, prev_window: Window):
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
