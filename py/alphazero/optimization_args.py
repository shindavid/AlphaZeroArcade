import argparse
from typing import Optional


class ModelingArgs:
    """
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
    minibatch_size: int
    snapshot_steps: int
    window_alpha: float
    window_beta: float
    window_c: int
    fixed_window_n: Optional[int]
    momentum: float
    weight_decay: float
    learning_rate: float

    @staticmethod
    def load(args):
        ModelingArgs.minibatch_size = args.minibatch_size
        ModelingArgs.snapshot_steps = args.snapshot_steps
        ModelingArgs.window_alpha = args.window_alpha
        ModelingArgs.window_beta = args.window_beta
        ModelingArgs.window_c = args.window_c
        ModelingArgs.fixed_window_n = args.fixed_window_n
        ModelingArgs.momentum = args.momentum
        ModelingArgs.weight_decay = args.weight_decay
        ModelingArgs.learning_rate = args.learning_rate

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('alphazero modeling options')

        group.add_argument('--minibatch-size', type=int, default=256,
                           help='minibatch size (default: %(default)s)')
        group.add_argument('--snapshot-steps', type=int, default=2048,
                           help='steps per snapshot (default: %(default)s)')
        group.add_argument('--window-alpha', type=float, default=0.75,
                           help='alpha for n_window formula (default: %(default)s)')
        group.add_argument('--window-beta', type=float, default=0.4,
                           help='beta for n_window formula (default: %(default)s)')
        group.add_argument('--window-c', type=int, default=250000,
                           help='c for n_window formula (default: %(default)s)')
        group.add_argument('--fixed-window-n', type=int,
                           help='if specified, uses this fixed window n instead of the n_window formula')
        group.add_argument('--momentum', type=float, default=0.9,
                           help='momentum (default: %(default)s)')
        group.add_argument('--weight-decay', type=float, default=6e-5,
                           help='weight decay (default: %(default)s)')
        group.add_argument('--learning-rate', type=float, default=6e-5,
                           help='learning rate (default: %(default)s)')
