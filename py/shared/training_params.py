import abc
import argparse
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional


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


VALID_WINDOW_SIZE_FUNCTIONS = {
    'katago': KataGoWindowSizeFunction,
    'fixed': FixedWindowSizeFunction,
}


@dataclass
class TrainingParams:
    """
    Parameters that control network training.

    Some notes on differences between AlphaGoZero and KataGo:

    AlphaGoZero used:

    - 64 GPU workers
    - 19 CPU parameter servers
    - Minibatch size of 32 per worker (32*64 = 2,048 in total)
    - Minibatches sampled uniformly randomly from most recent 500,000 games
    - Checkpointing every 1,000 training steps
    - Momentum of 0.9
    - L2 regularization parameter of 1e-4
    - Per-sample learning rate annealing from 10^-5 to 10^-7 (from 200k to 600k steps)
    - Unclear how they balanced policy loss vs value loss

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

    KataGo used:

    - 1 GPU
    - Minibatch size of 256
    - Minibatches sampled uniformly randomly from most recent f(N) samples (=positions, not games),
      where f(N) is the function g(x) = x^alpha shifted so that f(c)=c and f'(c)=beta, where
      (alpha, beta, c) = (0.75, 0.4, 250k)
    - Every ~250k training samples (~1000 training steps), weight snapshot is taken, and EMA of last
      4 snapshots with decay=0.75 is used as snapshot
    - Momentum of 0.9
    - L2 regularization parameter of 3e-5 (should correspond to weight_decay of 2*3e-5 = 6e-5)
    - Per-sample learning rate annealing from 2*10^-5 (first 5mil) to 6*10^-5, back down to 6*10^-6
      for the last day
    - Scaled value loss by 1.5

    https://arxiv.org/pdf/1902.10565.pdf

    TODO: weight EMA
    TODO: learning rate annealing
    """
    window_size_function_str: str = 'katago(n0=minibatches_per_epoch*minibatch_size)'
    target_sample_rate: float = 8
    minibatches_per_epoch: int = 2048
    minibatch_size: int = 256
    window_size_function: WindowSizeFunction = None

    def __post_init__(self):
        attrs = {f.name: getattr(self, f.name) for f in fields(self)}
        self.window_size_function = WindowSizeFunction.create(self.window_size_function_str, attrs)

    def samples_per_window(self):
        return self.minibatch_size * self.minibatches_per_epoch

    @staticmethod
    def create(args) -> 'TrainingParams':
        return TrainingParams(
            window_size_function_str=args.window_size_function,
            target_sample_rate=args.target_sample_rate,
            minibatches_per_epoch=args.minibatches_per_epoch,
            minibatch_size=args.minibatch_size,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: Optional['TrainingParams']=None):
        if defaults is None:
            defaults = TrainingParams()
        group = parser.add_argument_group('Learning options (defaults are game-specific)')

        valid_functions_str = str(list(VALID_WINDOW_SIZE_FUNCTIONS.keys()))
        group.add_argument('--window-size-function', default=defaults.window_size_function_str,
                           help=f'window size function (valid functions: {valid_functions_str}, default: "%(default)s")')
        group.add_argument(
            '--target-sample-rate', type=int, default=defaults.target_sample_rate,
            help='target number of times to train over a single row (default: %(default)s))')
        group.add_argument('--minibatches-per-epoch', type=int,
                           default=defaults.minibatches_per_epoch,
                           help='minibatches per epoch (default: %(default)s)')
        group.add_argument('--minibatch-size', type=int, default=defaults.minibatch_size,
                           help='minibatch size (default: %(default)s)')

    def add_to_cmd(self, cmd: List[str], defaults: Optional['TrainingParams']=None):
        if defaults is None:
            defaults = TrainingParams()
        if self.window_size_function_str != defaults.window_size_function_str:
            cmd.extend(['--window-size-function', self.window_size_function_str])
        if self.target_sample_rate != defaults.target_sample_rate:
            cmd.extend(['--target-sample-rate', str(self.target_sample_rate)])
        if self.minibatches_per_epoch != defaults.minibatches_per_epoch:
            cmd.extend(['--minibatches-per-epoch', str(self.minibatches_per_epoch)])
        if self.minibatch_size != defaults.minibatch_size:
            cmd.extend(['--minibatch-size', str(self.minibatch_size)])
