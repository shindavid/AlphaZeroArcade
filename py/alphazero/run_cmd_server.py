#!/usr/bin/env python3

import argparse
import os

from alphazero.logic.common_params import CommonParams
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.cmd_server import CmdServer, CmdServerParams
from alphazero.logic.sample_window_logic import SamplingParams
from util.logging_util import LoggingParams, configure_logger, get_logger


logger = get_logger()


def load_args():
    parser = argparse.ArgumentParser()

    CommonParams.add_args(parser)
    SamplingParams.add_args(parser)
    LoggingParams.add_args(parser)
    CmdServerParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    common_params = CommonParams.create(args)
    sampling_params = SamplingParams.create(args)
    logging_params = LoggingParams.create(args)
    params = CmdServerParams.create(args)

    log_filename = os.path.join(DirectoryOrganizer(common_params).logs_dir, 'cmd-server.log')
    configure_logger(filename=log_filename, params=logging_params)

    logger.info(f'**** Starting cmd-server ****')

    cmd_server = CmdServer(params, common_params, sampling_params)
    cmd_server.register_signal_handler()
    cmd_server.run()


if __name__ == '__main__':
    main()
