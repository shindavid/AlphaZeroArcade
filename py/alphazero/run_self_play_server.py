#!/usr/bin/env python3

"""
This script serves as a thin wrapper around a c++ binary. It communicates with the cmd server, and
upon receiving "start" requests from the cmd server, will start the c++ binary. From there, the
c++ binary and the cmd server communicate directly via TCP.

This setup allows us to relaunch the c++ binary process as needed under the hood of a single
self-play server process. This is useful because sometimes we want certain configuration changes
between generations. For example, gen-0 does not use a model, and in later generations we may want
to increase the number of MCTS simulations.
"""
import argparse
from dataclasses import dataclass

from alphazero.logic.common_params import CommonParams
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.self_play_server import SelfPlayServer, SelfPlayServerParams
from util.logging_util import LoggingParams, configure_logger, get_logger

import os


logger = get_logger()


def load_args():
    parser = argparse.ArgumentParser()

    CommonParams.add_args(parser)
    SelfPlayServerParams.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    common_params = CommonParams.create(args)
    params = SelfPlayServerParams.create(args)
    logging_params = LoggingParams.create(args)

    log_filename = os.path.join(DirectoryOrganizer(common_params).logs_dir, 'self-play-server.log')
    configure_logger(filename=log_filename, params=logging_params)

    logger.info(f'**** Starting self-play-server ****')

    server = SelfPlayServer(params, common_params)
    server.register_signal_handler()
    server.run()


if __name__ == '__main__':
    main()
