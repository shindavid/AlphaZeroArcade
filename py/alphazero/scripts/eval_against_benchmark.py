from alphazero.logic.benchmarker import DirectoryOrganizer
from alphazero.logic.evaluator import MCTSEvaluator
from alphazero.logic.run_params import RunParams
from util.logging_util import configure_logger
from util.py_util import CustomHelpFormatter

import argparse
import os
import shutil


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--benchmark_tag', type=str, default=100, help='run tag for benchmark')
    parser.add_argument('--target_eval_percent', type=float, default=0.1, help='target eval percent')
    parser.add_argument('--n_games', type=int, default=1000, help='Number of games per match')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of steps per evaluation')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    parser.add_argument('--error_threshold', type=float, default=100, help='error threshold for elo estimation')
    return parser.parse_args()

def main():
    configure_logger()
    args = load_args()
    run_params = RunParams.create(args)

    benchmark_tag = args.benchmark_tag
    run_params_benchmark = RunParams(run_params.game, benchmark_tag)
    benchmark_organizer = DirectoryOrganizer(run_params_benchmark, base_dir_root='/workspace')
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

    if not os.path.exists(organizer.eval_db_filename):
        shutil.copy2(benchmark_organizer.benchmark_db_filename, organizer.eval_db_filename(benchmark_tag))

    evaluator = MCTSEvaluator(organizer, benchmark_tag)
    evaluator.run(n_iters=args.n_iters, target_eval_percent=args.target_eval_percent,
                  n_games=args.n_games, error_threshold=args.error_threshold)

if __name__ == '__main__':
    main()

