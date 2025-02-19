from alphazero.logic.agent_types import Agent, MCTSAgent
from alphazero.logic.benchmarking import DirectoryOrganizer, BenchmarkCommittee
from alphazero.logic.evaluating import Evaluation
from alphazero.logic.run_params import RunParams

from tqdm import tqdm

def main():
    game = 'c4'
    tag = 'benchmark'
    benchmark_db_name = 'benchmark_i100'
    n_games = 100
    n_steps = 10

    organizer = DirectoryOrganizer(RunParams(game, tag))
    benchmark_committee = BenchmarkCommittee(organizer, db_name=benchmark_db_name,\
      load_past_data=True)
    benchmark_committee.compute_ratings()

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