#!/usr/bin/env python3

"""
This script serves as a thin wrapper around a c++ binary. It communicates with the loop controller,
and upon receiving "start" requests from the loop controller, will start the c++ binary. From there,
the c++ binary and the loop controller communicate directly via TCP.

This setup allows us to relaunch the c++ binary process as needed under the hood of a single
self-play server process. This is useful because sometimes we want certain configuration changes
between generations. For example, gen-0 does not use a model, and in later generations we may want
to increase the number of MCTS simulations.
"""
from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.servers.gaming.self_play_server import SelfPlayServer, SelfPlayServerParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo

import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    SelfPlayServerParams.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    params = SelfPlayServerParams.create(args)
    docker_params = DockerParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    os.chdir(Repo.root())

    # The logger will actually be reconfigured later in the server. We have this call here to cover
    # any logging that happens before that point. Also, configure_logger() contains a key piece of
    # logic to prevent logging race-condition deadlocks, which needs to be set up before any threads
    # are spawned, making it important to call this here.
    configure_logger(params=logging_params)

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    server = SelfPlayServer(params, logging_params, build_params)
    server.run()


if __name__ == '__main__':
    main()
