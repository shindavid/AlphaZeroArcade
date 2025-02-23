from alphazero.logic.benchmarker import DirectoryOrganizer
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.run_params import RunParams
from util.logging_util import configure_logger
from util.py_util import CustomHelpFormatter

from tqdm import tqdm
import argparse

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--benchmark_tag', type=str, default=100, help='run tag for benchmark')
    parser.add_argument('--target_eval_percent', type=float, default=1.0, help='target eval percent')
    parser.add_argument('--n_games', type=int, default=1000, help='Number of games per match')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of steps per evaluation')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    parser.add_argument('--max_version_gap', type=int, default=500, help='used for initial rating estimate')
    return parser.parse_args()

def main():
    configure_logger()
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = load_args()
    run_params = RunParams.create(args)

    benchmark_tag = args.benchmark_tag
    run_params_benchmark = RunParams(run_params.game, benchmark_tag)
    benchmark_organizer = DirectoryOrganizer(run_params_benchmark, base_dir_root='/workspace')
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

    evaluator = Evaluator(organizer, benchmark_organizer)
    evaluator.run(n_iters=args.n_iters, target_eval_percent=args.target_eval_percent,
                  n_games=args.n_games, max_version_gap=args.max_version_gap)

if __name__ == '__main__':
    main()

