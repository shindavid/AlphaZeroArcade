#!/usr/bin/env python3

"""
This script serves as a thin wrapper around a c++ binary. It communicates with the loop controller,
and upon receiving "start" requests from the loop controller, will start the c++ binary. From there,
the c++ binary and the loop controller communicate directly via TCP.

This setup allows us to relaunch the c++ binary process as needed under the hood of a single
ratings server process. This is useful because it allows us to launch each matchup requested by
the loop controller as a separate c++ binary process.
"""
from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.servers.gaming.ratings_server import RatingsServer, RatingsServerParams
from shared.rating_params import RatingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo

import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    RatingsServerParams.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser)
    RatingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    params = RatingsServerParams.create(args)
    docker_params = DockerParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)
    rating_params = RatingParams.create(args)

    os.chdir(Repo.root())

    # The logger will actually be reconfigured later in the server. We have this call here to cover
    # any logging that happens before that point. Also, configure_logger() contains a key piece of
    # logic to prevent logging race-condition deadlocks, which needs to be set up before any threads
    # are spawned, making it important to call this here.
    configure_logger(params=logging_params)

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    server = RatingsServer(params, logging_params, build_params, rating_params)
    server.run()


if __name__ == '__main__':
    main()
