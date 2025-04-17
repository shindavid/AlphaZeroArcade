from alphazero.logic.benchmarker import  Benchmarker
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter
from util.logging_util import LoggingParams, configure_logger

import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    LoggingParams.add_args(parser)
    parser.add_argument('--target_elo_gap', type=int, default=100, help='target elo gap')
    parser.add_argument('--n_games', type=int, default=100, help='Number of games per match')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')

    args = parser.parse_args()

    logging_params = LoggingParams.create(args)
    log_filename = '/workspace/repo/benchmarker.log'
    configure_logger(filename=log_filename, params=logging_params, mode='a')

    run_params = RunParams.create(args)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

    benchmarker = Benchmarker(organizer)
    benchmarker.run(n_iters=args.n_iters,
                            n_games=args.n_games,
                            target_elo_gap=args.target_elo_gap)


if __name__ == '__main__':
    main()
