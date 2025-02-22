from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.benchmarker import DirectoryOrganizer, Benchmarker
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.run_params import RunParams
from util.py_util import CustomHelpFormatter

from tqdm import tqdm
import argparse

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--benchmark_tag', type=int, default=100, help='run tag for benchmark')
    parser.add_argument('--n_games', type=int, default=100, help='Number of games per match')
    parser.add_argument("-i", '--n_iters', type=int, default=100, help='Number of MCTS iterations')
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = load_args()
    run_params = RunParams.create(args)

    benchmark_tag = args.benchmark_tag
    run_params_benchmark = RunParams(run_params.game, benchmark_tag)
    benchmark_organizer = DirectoryOrganizer(run_params_benchmark)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

    evaluator = Evaluator(organizer, benchmark_organizer)
    evaluator.run(n_iters=args.n_iters)
    eval_tag = 'benchmark'
    eval_db_name = 'test_eval'
    eval_organizer = DirectoryOrganizer(RunParams(game, eval_tag))
    evaluation = Evaluation(eval_organizer, benchmark_committee, eval_db_name)
    test_agents = [MCTSAgent(gen=32, n_iters=1, organizer=eval_organizer)]
    for test_agent in tqdm(test_agents):
        test_rating = evaluation.evaluate(test_agent, n_games=n_games, n_steps=n_steps)
        print(f'{test_agent}: {test_rating}')

if __name__ == '__main__':
    main()

