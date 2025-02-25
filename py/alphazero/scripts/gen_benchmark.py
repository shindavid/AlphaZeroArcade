from alphazero.logic.agent_types import MCTSAgent, ReferenceAgent
from alphazero.logic.benchmarker import  Benchmarker
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.match_runner import MatchRunner, Match
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter
from util.logging_util import get_logger, configure_logger

from typing import Optional
import argparse

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--target_elo_gap', type=int, default=100, help='target elo gap')
    parser.add_argument('--n_games', type=int, default=100, help='Number of games per match')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    return parser.parse_args()

def main():
    configure_logger()
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = load_args()
    run_params = RunParams.create(args)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    benchmarker = Benchmarker(organizer)
    benchmarker.run(n_iters=args.n_iters,
                            n_games=args.n_games,
                            target_elo_gap=args.target_elo_gap,)
    evaluator = Evaluator(organizer, organizer)
    ref_agents =[ReferenceAgent(type_str='Perfect',
                                strength_param='strength',
                                strength=strength,
                                binary_filename=organizer.binary_filename) \
                                    for strength in range(1, 22, 5)]
    for ref_agent in ref_agents:
        evaluator.eval_agent(ref_agent, n_games=1000)


if __name__ == '__main__':
    main()

