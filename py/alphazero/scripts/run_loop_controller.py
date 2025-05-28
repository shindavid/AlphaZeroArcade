#!/usr/bin/env python3
from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.loop_controller import LoopController, LoopControllerParams
from games.game_spec import GameSpec
from shared.rating_params import RatingParams
from shared.training_params import TrainingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo

import argparse
import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_training_params = None if game_spec is None else game_spec.training_params
    default_rating_params = None if game_spec is None else game_spec.rating_params
    LoopControllerParams.add_args(parser)
    TrainingParams.add_args(parser, defaults=default_training_params)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser, loop_controller=True)
    RatingParams.add_args(parser, defaults=default_rating_params)

    return parser.parse_args()


def main():
    args = load_args()
    run_params = RunParams.create(args)
    params = LoopControllerParams.create(args)
    training_params = TrainingParams.create(args)
    docker_params = DockerParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)
    rating_params = RatingParams.create(args)

    os.chdir(Repo.root())

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    if not organizer.version_check():
        print('The following output directory is outdated:\n')
        print(organizer.base_dir + '\n')
        print('As a result, you cannot resume a run from this directory.')
        print('Please try again with a new tag.')
        return

    server = LoopController(params, training_params, run_params, build_params, rating_params)
    log_filename = os.path.join(server.organizer.logs_dir, 'loop-controller.log')

    if server.on_ephemeral_local_disk_env:
        # first copy over the loop-controller.log from persistent to scratch. This makes it so
        # rsyncing back from scraptch to persistent effectively acts as an append operation.
        backup = os.path.join(server.persistent_organizer.logs_dir, 'loop-controller.log')
        if os.path.isfile(backup):
            os.system(f'cp {backup} {log_filename}')

    configure_logger(filename=log_filename, params=logging_params, mode='a')
    logger.info('**** Starting loop-controller ****')

    server.run()


if __name__ == '__main__':
    main()
