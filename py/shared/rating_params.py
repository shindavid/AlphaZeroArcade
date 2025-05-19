from dataclasses import dataclass, field

import argparse
from typing import List, Optional

@dataclass
class DefaultTargetEloGap:
    first_run: float = 500.0
    benchmark: float = 100.0

    def __post_init__(self):
        assert self.first_run > 0, "ELO gap must be positive"
        assert self.benchmark > 0, "ELO gap must be positive"
        assert self.first_run >= self.benchmark, "First run ELO gap must be >= benchmark ELO gap"


@dataclass
class RatingPlayerOptions:
    num_search_threads: int = 4
    num_iterations: int = 100

    def __post_init__(self):
        assert self.num_search_threads > 0, "Must have >0 search threads"
        assert self.num_iterations > 0, "Must have >0 iterations"


@dataclass
class RatingParams:
    """
    Holds parameters for rating processes such as benchmarking and evaluation.
    """
    rating_player_options: RatingPlayerOptions = field(default_factory=RatingPlayerOptions)
    default_target_elo_gap: DefaultTargetEloGap = field(default_factory=DefaultTargetEloGap)

    eval_error_threshold: float = 50.0
    n_games_per_benchmark: int = 100
    n_games_per_evaluation: int = 1000

    target_elo_gap: Optional[float] = None
    use_remote_play: bool = False

    def __post_init__(self):
        assert self.n_games_per_benchmark > 0, "Must have >0 games per benchmark"
        assert self.n_games_per_evaluation > 0, "Must have >0 games per evaluation"
        if self.target_elo_gap is not None:
            assert self.target_elo_gap > 0, "ELO gap must be positive"

    @staticmethod
    def create(args) -> 'RatingParams':
        return RatingParams(
            rating_player_options=RatingPlayerOptions(
                num_search_threads=args.num_search_threads,
                num_iterations=args.num_iterations,
                ),
            default_target_elo_gap=DefaultTargetEloGap(
              first_run=args.first_run_target_elo_gap,
              benchmark=args.benchmark_target_elo_gap,
              ),
            target_elo_gap=args.target_elo_gap,
            eval_error_threshold=args.eval_error_threshold,
            n_games_per_benchmark=args.n_games_per_benchmark,
            n_games_per_evaluation=args.n_games_per_evaluation,
            use_remote_play=args.use_remote_play,
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: Optional['RatingParams']=None):
        if defaults is None:
            defaults = RatingParams()
        group = parser.add_argument_group('Rating options')

        group.add_argument('--num-search-threads', type=int, default=defaults.rating_player_options.num_search_threads,
                           help='number of search threads for the rating player (default: %(default)s)')
        group.add_argument('--num-iterations', type=int, default=defaults.rating_player_options.num_iterations,
                           help='number of MCTS iterations for the rating player (default: %(default)s)')
        group.add_argument('--first-run-target-elo-gap', type=float, default=defaults.default_target_elo_gap.first_run,
                           help='target elo gap for first run (default: %(default).1f)')
        group.add_argument('--benchmark-target-elo-gap', type=float, default=defaults.default_target_elo_gap.benchmark,
                           help='target elo gap for benchmark (default: %(default).1f)')
        group.add_argument('--target-elo-gap', type=float, default=None,
                           help='target elo gap for rating (default: None)')
        group.add_argument('--eval-error-threshold', type=float, default=defaults.eval_error_threshold,
                           help='error threshold for Elo estimation (default: %(default).1f)')
        group.add_argument('--n-games-per-benchmark', type=int, default=defaults.n_games_per_benchmark,
                           help='number of games per benchmark (default: %(default)d)')
        group.add_argument('--n-games-per-evaluation', type=int, default=defaults.n_games_per_evaluation,
                           help='number of games per evaluation (default: %(default)d)')
        group.add_argument('--use-remote-play', action='store_true', default=defaults.use_remote_play,
                           help='use remote play (multiple binaries).')


    def add_to_cmd(self, cmd: List[str], loop_controller=False):
        defaults = RatingParams()
        if loop_controller:
            if self.rating_player_options.num_iterations != defaults.rating_player_options.num_iterations:
                cmd.extend(['--num-iterations', str(self.rating_player_options.num_iterations)])
        else:
            if self.rating_player_options.num_search_threads != defaults.rating_player_options.num_search_threads:
                cmd.extend(['--num-search-threads', str(self.rating_player_options.num_search_threads)])

