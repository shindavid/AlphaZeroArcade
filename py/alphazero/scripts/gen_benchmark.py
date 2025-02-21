from alphazero.logic.agent_types import MCTSAgent, ReferenceAgent
from alphazero.logic.benchmarker import  Benchmarker
from alphazero.logic.match_runner import MatchRunner, Match
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter

from typing import Optional
import argparse

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--n_games', type=int, default=100, help='Number of games per match')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = load_args()
    run_params = RunParams.create(args)
    organizer = DirectoryOrganizer(run_params)
    benchmark_committee = Benchmarker(organizer, load_past_data=True)
    benchmark_committee.run()


if __name__ == '__main__':
    main()

