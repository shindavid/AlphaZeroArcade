#!/usr/bin/env python3

import argparse

from alphazero.logic.build_params import BuildParams
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.loop_controller import LoopController, LoopControllerParams
from games.game_spec import GameSpec
from shared.training_params import TrainingParams
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.py_util import CustomHelpFormatter

import os
from typing import Optional


logger = get_logger()


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_training_params = None if game_spec is None else game_spec.training_params
    LoopControllerParams.add_args(parser)
    TrainingParams.add_args(parser, defaults=default_training_params)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser, add_binary_path_option=False)

    return parser.parse_args()


def main():
    args = load_args()
    run_params = RunParams.create(args)
    params = LoopControllerParams.create(args)
    training_params = TrainingParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    log_filename = os.path.join(DirectoryOrganizer(run_params).logs_dir, 'loop-controller.log')
    configure_logger(filename=log_filename, params=logging_params, mode='a')

    logger.info('**** Starting loop-controller ****')

    server = LoopController(params, training_params, run_params, build_params)
    server.run()


if __name__ == '__main__':
    main()
