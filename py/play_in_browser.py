#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter

import argparse
import os
import subprocess


def get_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    RunParams.add_args(parser, tag_help_str='default: most recently created tag')
    parser.add_argument("-d", '--debug', action='store_true', help='debug build')
    parser.add_argument("-i", '--num-mcts-iters', type=int, default=100,
                        help='num MCTS iterations (default: %(default)s)')
    parser.add_argument("-m", '--model-generation', type=int,
                        help='model generation (default: latest)')
    return parser.parse_args()


def main():
    args = get_args()

    run_params = RunParams.create(args, require_tag=False)
    game = run_params.game

    if not run_params.tag:
        run_params.tag = DirectoryOrganizer.find_latest_tag(game, Workspace)
        if not run_params.tag:
            raise Exception(f'No tags found for game {game} in {Workspace.benchmark_dir}')
        print(f'Using latest tag: {run_params.tag}')

    organizer = DirectoryOrganizer(run_params, Workspace)
    gen = args.model_generation
    if gen is None:
        gen = organizer.get_latest_model_generation()
        if gen is None:
            raise Exception(f'No model generations found for {game}/{run_params.tag} in '
                            f'{Workspace.output_dir()}')
        print(f'Using latest gen: {gen}')
    model_filename = organizer.get_model_filename(gen)
    if not os.path.exists(model_filename):
        raise Exception(f'Model file does not exist: {model_filename}')

    build_type = 'Debug' if args.debug else 'Release'
    bin = f'target/{build_type}/bin/{game}'

    num_mcts_iters = args.num_mcts_iters

    # cmd:
    #
    # {bin} --player "--type=web" --player "--type=MCTS-C -m {model_filename} -i {num_mcts_iters}"

    cmd = [
        bin,
        '--player', '"--type=web"',
        '--player', f'"--type=MCTS-C -m {model_filename} -i {num_mcts_iters}"'
    ]

    cmd = ' '.join(cmd)
    print(f'Running: {cmd}\n')
    proc = subprocess.Popen(cmd, shell=True, text=True, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    if proc.returncode:
        print(stderr)


if __name__ == '__main__':
    main()
