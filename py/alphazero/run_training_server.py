#!/usr/bin/env python3

import argparse

from alphazero.logic.common_params import CommonParams
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.training_server import TrainingServer, TrainingServerParams
from alphazero.logic.learning_params import LearningParams
from alphazero.logic.sample_window_logic import SamplingParams
from util.logging_util import LoggingParams, configure_logger, get_logger

import os


logger = get_logger()


def load_args():
    parser = argparse.ArgumentParser()

    CommonParams.add_args(parser)
    SamplingParams.add_args(parser)
    TrainingServerParams.add_args(parser)
    LearningParams.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    common_params = CommonParams.create(args)
    params = TrainingServerParams.create(args)
    sampling_params = SamplingParams.create(args)
    learning_params = LearningParams.create(args)
    logging_params = LoggingParams.create(args)

    log_filename = os.path.join(DirectoryOrganizer(common_params).logs_dir, 'training-server.log')
    configure_logger(filename=log_filename, params=logging_params)

    logger.info(f'**** Starting training-server ****')

    server = TrainingServer(params, learning_params, sampling_params, common_params)
    server.register_signal_handler()
    server.run()


if __name__ == '__main__':
    main()
