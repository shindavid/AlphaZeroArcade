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
import argparse

from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.servers.gaming.self_play_server import SelfPlayServer, SelfPlayServerParams
from util.logging_util import LoggingParams
from util.py_util import CustomHelpFormatter


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    SelfPlayServerParams.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser, add_ffi_lib_path_option=False)

    return parser.parse_args()


def main():
    args = load_args()
    params = SelfPlayServerParams.create(args)
    docker_params = DockerParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    server = SelfPlayServer(params, logging_params, build_params)
    server.run()


if __name__ == '__main__':
    main()
