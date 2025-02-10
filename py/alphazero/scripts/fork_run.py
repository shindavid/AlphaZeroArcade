#!/usr/bin/env python3

"""
fork_run.py effectively copies a run-directory to a new directory.

Usage:

./fork_run.py -g GAME -f FROM_TAG -t TO_TAG
"""

from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.loop_controller import LoopController, LoopControllerParams
from shared.training_params import TrainingParams
import games.index as game_index
from util.logging_util import LoggingParams, configure_logger, get_logger

import argparse
import os
import sys


logger = get_logger()


def load_args():
    description = """fork_run.py effectively copies a run-directory to a new directory.

By default, this does a "soft fork", meaning that the new run will soft-link to the self-play
data files of the previous run. If --hard-fork is specified, the self-play data files are copied.

Noteworthy options:

--last-gen:
    Ignore any data that comes after this generation.

--retrain-models:
    Do not copy model files. The forked run will retrain the models using the same self-play data
    and training-windows as the previous run. This is useful for doing supervised-learning
    experiments, such as trying different network architectures or training hyperparameters.
"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    group = parser

    game_index.add_parser_argument(group, '-g', '--game')
    group.add_argument('-f', '--from-tag', help='tag to fork from')
    group.add_argument('-t', '--to-tag', help='tag to fork to')
    group.add_argument('--hard-fork', action='store_true',
                       help='copies self-play data from the previous run. By default, '
                       'the new run will still point to the previous run for self-play data.')
    group.add_argument('--last-gen', type=int,
                       help='Only copy models up to this generation, along with data created by '
                       'those models')
    group.add_argument('--retrain-models', action='store_true',
                       help='Do not copy model files. The forked run will retrain the models '
                       'using the same self-play data and training-windows as the previous run.')

    return parser.parse_args()


def main():
    args = load_args()
    configure_logger()

    if args.from_tag is None:
        raise ValueError('Required option: --from-tag/-f')
    if args.to_tag is None:
        raise ValueError('Required option: --to-tag/-t')
    if args.game is None:
        raise ValueError('Required option: --game/-g')

    assert game_index.is_valid_game_name(args.game), f'Invalid game name: {args.game}'

    from_params = RunParams(args.game, args.from_tag)
    to_params = RunParams(args.game, args.to_tag)

    from_organizer = DirectoryOrganizer(from_params, base_dir_root='/workspace')
    to_organizer = DirectoryOrganizer(to_params, base_dir_root='/workspace')

    if not os.path.isdir(from_organizer.base_dir):
        raise ValueError(f'From-directory does not exist: {from_organizer.base_dir}')
    if from_organizer.fork_info is not None:
        raise ValueError(f'From-directory is already a fork: {from_organizer.base_dir}')
    if os.path.isdir(to_organizer.base_dir):
        raise ValueError(f'To-directory already exists: {to_organizer.base_dir}')

    hard_fork = bool(args.hard_fork)
    retrain_models = bool(args.retrain_models)
    last_gen = args.last_gen

    if last_gen is not None:
        if last_gen <= 0:
            raise ValueError('last-self-play-gen must be greater than 0')

    to_organizer.makedirs()

    if hard_fork:
        logger.info('Copying self-play data...')
        from_organizer.copy_self_play_data(to_organizer, last_gen - 1)

    if retrain_models:
        logger.info('Skipping model files...')
    else:
        logger.info('Copying model files...')
        from_organizer.copy_models_and_checkpoints(to_organizer, last_gen)

    logger.info('Copying database files...')
    from_organizer.copy_databases(to_organizer, retrain_models, last_gen)

    logger.info('Writing fork info...')
    to_organizer.write_fork_info(from_organizer, hard_fork, retrain_models, last_gen)

    logger.info('Fork complete!')


if __name__ == '__main__':
    main()
