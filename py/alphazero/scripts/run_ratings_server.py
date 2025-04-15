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
from alphazero.logic.custom_types import ClientRole
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.servers.gaming.ratings_server import RatingsServer, RatingsServerParams
from alphazero.servers.gaming.server_base import ServerConfig
from util.logging_util import LoggingParams
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo

import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    RatingsServerParams.add_args(parser, server_name='ratings-server')
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    params = RatingsServerParams.create(args)
    docker_params = DockerParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    os.chdir(Repo.root())

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    server_config = ServerConfig(
        server_name='benchmark-server',
        worker_name='benchmark-worker',
        server_role=ClientRole.BENCHMARK_SERVER,
        worker_role=ClientRole.BENCHMARK_WORKER)

    server = RatingsServer(params, logging_params, build_params, server_config)
    server.run()


if __name__ == '__main__':
    main()
