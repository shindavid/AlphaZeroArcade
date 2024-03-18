#!/usr/bin/env python3

import argparse

from alphazero.logic.common_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.loop_controller import LoopController, LoopControllerParams
from alphazero.logic.training_params import TrainingParams
from util.logging_util import LoggingParams, configure_logger, get_logger

import os


logger = get_logger()


def load_args():
    parser = argparse.ArgumentParser()

    RunParams.add_args(parser)
    LoopControllerParams.add_args(parser)
    TrainingParams.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    run_params = RunParams.create(args)
    params = LoopControllerParams.create(args)
    training_params = TrainingParams.create(args)
    logging_params = LoggingParams.create(args)

    log_filename = os.path.join(DirectoryOrganizer(run_params).logs_dir, 'loop-controller.log')
    configure_logger(filename=log_filename, params=logging_params)

    logger.info(f'**** Starting loop-controller ****')

    server = LoopController(params, training_params, run_params)
    server.run()


if __name__ == '__main__':
    main()
