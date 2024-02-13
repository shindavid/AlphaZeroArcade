import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class LearningParams:
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
    momentum: float = 0.9
    weight_decay: float = 6e-5
    learning_rate: float = 6e-5

    @staticmethod
    def create(args) -> 'LearningParams':
        return LearningParams(
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        defaults = LearningParams()
        group = parser.add_argument_group('Learning options')

        group.add_argument('--momentum', type=float, default=defaults.momentum,
                           help='momentum (default: %(default)s)')
        group.add_argument('--weight-decay', type=float, default=defaults.weight_decay,
                           help='weight decay (default: %(default)s)')
        group.add_argument('--learning-rate', type=float, default=defaults.learning_rate,
                           help='learning rate (default: %(default)s)')

    def add_to_cmd(self, cmd: List[str]):
        defaults = LearningParams()
        if self.momentum != defaults.momentum:
            cmd.extend(['--momentum', str(self.momentum)])
        if self.weight_decay != defaults.weight_decay:
            cmd.extend(['--weight-decay', str(self.weight_decay)])
        if self.learning_rate != defaults.learning_rate:
            cmd.extend(['--learning-rate', str(self.learning_rate)])
