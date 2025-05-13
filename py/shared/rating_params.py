from dataclasses import dataclass

import argparse
from typing import Optional

@dataclass
class TargetEloGap:
    first_run: float = 500.0
    benchmark: float = 100.0


@dataclass
class RatingPlayerOptions:
    num_search_threads: int = 4
    num_iterations: int = 100


@dataclass
class RatingParams:
    rating_player_options: RatingPlayerOptions = RatingPlayerOptions()
    target_elo_gap: TargetEloGap = TargetEloGap()
    eval_error_threshold: float = 50.0
    n_games_per_benchmark: int = 100
    n_games_per_evaluation: int = 1000

    @staticmethod
    def create(args) -> 'RatingParams':
        return RatingParams(
            rating_player_options=RatingPlayerOptions(
                num_search_threads=args.num_search_threads,
                num_iterations=args.num_iterations
            ),
            target_elo_gap=TargetEloGap(
                first_run=args.first_run_target_elo_gap,
                benchmark=args.benchmark_target_elo_gap
            ),
            eval_error_threshold=args.eval_error_threshold,
            n_games_per_benchmark=args.n_games_per_benchmark,
            n_games_per_evaluation=args.n_games_per_evaluation
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'RatingParams'):
        group = parser.add_argument_group('Rating options')
        group.add_argument('-n', '--num-search-threads', type=int, default=defaults.rating_player_options.num_search_threads,
                           help='number of search threads for the rating player (default: %(default)s)')
        group.add_argument('-i', '--num-iterations', type=int, default=defaults.rating_player_options.num_iterations,
                           help='number of MCTS iterations for the rating player (default: %(default)s)')
        group.add_argument('--first-run-target-elo-gap', type=float, default=defaults.target_elo_gap.first_run,
                           help='target elo gap for first run (default: %(default).1f)')
        group.add_argument('--benchmark-target-elo-gap', type=float, default=defaults.target_elo_gap.benchmark,
                           help='target elo gap for benchmark (default: %(default).1f)')
        group.add_argument('--eval-error-threshold', type=float, default=defaults.eval_error_threshold,
                           help='error threshold for Elo estimation (default: %(default).1f)')
        group.add_argument('--n-games-per-benchmark', type=int, default=defaults.n_games_per_benchmark,
                           help='number of games per benchmark (default: %(default)d)')
        group.add_argument('--n-games-per-evaluation', type=int, default=defaults.n_games_per_evaluation,
                           help='number of games per evaluation (default: %(default)d)')
