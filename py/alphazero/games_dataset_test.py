#!/usr/bin/env python3
"""
A simple script to test differently parameterized usages of GamesDatasetGenerator.

After some ad-hoc experimentation, I like the following:

- At each generation, sample M = 2**19 = 524288 positions.
- At generation N, take that sample from the first M * sqrt(N) positions in the master list.
- If any position in the master list becomes sampled more than K = 4 times in expectation, discard
  it, and all preceding positions, from the master list.
"""
import argparse
import os

from config import Config
from alphazero.data.position_dataset import GamesDatasetGenerator


class Args:
    alphazero_dir: str
    game: str
    tag: str
    sample_limit: int
    num_positions_to_sample_per_generation: int
    policy: str

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tag = args.tag
        Args.sample_limit = args.sample_limit
        Args.num_positions_to_sample_per_generation = args.num_positions_to_sample_per_generation
        Args.policy = args.policy


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-g', '--game', help='the game')
    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('-s', '--sample-limit', type=int, default=4,
                        help='discard positions that have been sampled this many times in '
                        'expectation (default: %(default)s)')
    parser.add_argument('-n', '--num-positions-to-sample-per-generation', type=int, default=2**19,
                        help='number of positions to sample per generation (default: %(default)s)')
    parser.add_argument('-p', '--policy', choices=['prefix', 'suffix'], default='prefix',
                        help='policy for sampling positions, prefix or suffix '
                        '(default: %(default)s)')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')

    args = parser.parse_args()
    Args.load(args)


def main():
    load_args()

    sample_limit = Args.sample_limit
    use_prefix = Args.policy == 'prefix'
    num_positions_to_sample_per_generation = Args.num_positions_to_sample_per_generation
    loader_size = num_positions_to_sample_per_generation

    self_play_data_dir = os.path.join(Args.alphazero_dir, Args.game, Args.tag, 'self-play-data')
    generator = GamesDatasetGenerator(self_play_data_dir, sample_limit)
    generation = 0

    print(f'Generation {generation}...')
    while True:
        dataset = generator.get_next_dataset(loader_size, use_prefix, verbose=True)
        if dataset is None:
            break

        generator.record_dataset_usage(dataset, num_positions_to_sample_per_generation)
        # loader_size = int(loader_size * 1.03)
        generation += 1
        loader_size = int(num_positions_to_sample_per_generation * (generation + 1) ** 0.5)
        print(f'\nGeneration {generation}...')



if __name__ == '__main__':
    main()
