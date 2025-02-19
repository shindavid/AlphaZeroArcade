from alphazero.logic.benchmarking import  BenchmarkCommittee
from alphazero.logic.match_runner import MatchRunner
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from util.py_util import CustomHelpFormatter

from typing import Optional
import argparse

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--db_name', type=str, default='benchmark', help='Database name')
    parser.add_argument('--n_games', type=int, default=100, help='Number of games per match')
    parser.add_argument('--gen_start', type=int, default=0, help='Starting generation for match scheduling')
    parser.add_argument('--gen_end', type=int, default=128, help='Ending generation for match scheduling')
    parser.add_argument('--freq', type=int, default=4, help='Frequency of picking generations for match scheduling')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = load_args()
    run_params = RunParams.create(args)
    n_games = args.n_games
    organizer = DirectoryOrganizer(run_params)
    benchmark_committee = BenchmarkCommittee(organizer, args.db_name, load_past_data=True)
    matches = MatchRunner.linspace_matches(args.gen_start, args.gen_end, n_iters=args.n_iters, freq=args.freq, n_games=n_games, \
        organizer=organizer)
    benchmark_committee.play_matches(matches)
    ratings = benchmark_committee.compute_ratings()
    print(ratings)

if __name__ == '__main__':
    main()

