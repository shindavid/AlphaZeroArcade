from alphazero.logic.agent_types import ReferenceAgent
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.index import get_game_spec
from util.logging_util import configure_logger
from util.py_util import CustomHelpFormatter

import argparse
import os
import shutil

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--n_games', type=int, default=1000, help='Number of games per match')
    parser.add_argument('--error_threshold', type=float, default=100, help='error threshold for elo estimation')


    return parser.parse_args()


def main():
    configure_logger()

    args = load_args()
    run_params = RunParams.create(args)
    game_spec = get_game_spec(run_params.game)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

    if not os.path.exists(organizer.eval_db_filename):
        shutil.copy2(organizer.benchmark_db_filename, organizer.eval_db_filename)

    family = game_spec.reference_player_family
    type_str = family.type_str
    strength_param = family.strength_param
    ref_agents = []
    for strength in range(1, 22, 5):
        ref_agents.append(ReferenceAgent(type_str, strength_param, strength, tag=organizer.tag))

    evaluator = Evaluator(organizer)
    for agent in ref_agents:
        evaluator.eval_agent(agent, n_games=args.n_games,
                             error_threshold=args.error_threshold)


if __name__ == '__main__':
    main()

