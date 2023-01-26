import argparse
from dataclasses import dataclass
from typing import Any


@dataclass
class Param:
    short_name: str
    long_name: str
    value: Any
    help: str

    @property
    def value_type(self):
        return type(self.value)


class OptimizationArgParams:
    """
    AlphaGoZero optimization used:

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
    - Minibatches sampled uniformly randomly from most recent f(N) samples (=positions, not games), where f(N) is
      the function g(x) = x^alpha shifted so that f(c)=c and f'(c)=beta, where (alpha, beta, c) = (0.75, 0.4, 250k)
    - Every ~250k training samples (~1000 training steps), weight snapshot is taken, and EMA of last 4 snapshots
      with decay=0.75 is used as snapshot
    - Momentum of 0.9
    - L2 regularization parameter of 3e-5 (should correspond to weight_decay of 2*3e-5 = 6e-5)
    - Per-sample learning rate annealing from 2*10^-5 (first 5mil) to 6*10^-5, back down to 6*10^-6 for the last day
    - Scaled value loss by 1.5

    https://arxiv.org/pdf/1902.10565.pdf

    TODO: weight EMA
    TODO: learning rate annealing
    """
    minibatch_size = Param('-m', '--minibatch-size', 256, 'minibatch size')
    snapshot_steps = Param('-s', '--snapshot-steps', 1024, 'steps per snapshot')
    window_alpha = Param('-A', '--window-alpha', 0.75, 'alpha for n_window formula')
    window_beta = Param('-B', '--window-beta', 0.4, 'beta for n_window formula')
    window_c = Param('-c', '--window-c', 250000, 'c for n_window formula')
    momentum = Param('-M', '--momentum', 0.9, 'momentum')
    weight_decay = Param('-w', '--weight-decay', 6e-5, 'weight decay')
    learning_rate = Param('-l', '--learning-rate', 6e-5, 'learning rate')


class OptimizationArgs:
    attrs = [attr for attr in dir(OptimizationArgParams) if isinstance(getattr(OptimizationArgParams, attr), Param)]

    minibatch_size: int
    snapshot_steps: int
    window_alpha: float
    window_beta: float
    window_c: int
    momentum: float
    weight_decay: float
    learning_rate: float

    @staticmethod
    def load(args):
        for attr in OptimizationArgs.attrs:
            setattr(OptimizationArgs, attr, getattr(args, attr))

    @staticmethod
    def get_str() -> str:
        tokens = []

        for attr in OptimizationArgs.attrs:
            param: Param = getattr(OptimizationArgParams, attr)
            current = getattr(OptimizationArgs, attr)
            if current != param.value:
                tokens.extend([param.short_name, str(current)])

        return ' '.join(tokens)


def add_optimization_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('alphazero optimization options')

    for attr in OptimizationArgs.attrs:
        param: Param = getattr(OptimizationArgParams, attr)
        group.add_argument(param.short_name, param.long_name, type=param.value_type, default=param.value,
                           help=f'{param.help} (default: %(default)s)')
