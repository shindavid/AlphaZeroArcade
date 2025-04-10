from alphazero.logic.benchmarker import  Benchmarker
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter
from util.logging_util import LoggingParams, configure_logger

import argparse


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--target_elo_gap', type=int, default=100, help='target elo gap')
    parser.add_argument('--n_games', type=int, default=100, help='Number of games per match')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    parser.add_argument('--debug', action='store_true', help='enable debug logging')
    parser.add_argument('--debug-module', type=str, nargs='+', default=[],
                        help='specific module(s) to enable debug logging for. Example: '
                             '--debug-module=util.sqlite3_util --debug-module=alphazero.servers.gaming.session_data')
    return parser.parse_args()


def main():

    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = load_args()

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
