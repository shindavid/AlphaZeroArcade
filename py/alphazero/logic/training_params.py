import argparse
from typing import List


class TrainingParams:
    """
    Parameters that control network training.

    Sampling-related parameters could live here, but instead live in SamplingParams. This
    separation reflects the separation of responsibilities between the cmd-server and the
    training-server.

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
    momentum: float
    weight_decay: float
    learning_rate: float

    @staticmethod
    def load(args):
        TrainingParams.momentum = args.momentum
        TrainingParams.weight_decay = args.weight_decay
        TrainingParams.learning_rate = args.learning_rate

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Training options')

        group.add_argument('--momentum', type=float, default=0.9,
                           help='momentum (default: %(default)s)')
        group.add_argument('--weight-decay', type=float, default=6e-5,
                           help='weight decay (default: %(default)s)')
        group.add_argument('--learning-rate', type=float, default=6e-5,
                           help='learning rate (default: %(default)s)')

    @staticmethod
    def add_to_cmd(cmd: List[str]):
        cmd.extend([
            '--momentum', str(TrainingParams.momentum),
            '--weight-decay', str(TrainingParams.weight_decay),
            '--learning-rate', str(TrainingParams.learning_rate),
        ])
