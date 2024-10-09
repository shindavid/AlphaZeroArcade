#!/usr/bin/env python3

"""
This script serves as a thin wrapper around a c++ binary. It communicates with the loop controller,
and upon receiving "start" requests from the loop controller, will start the c++ binary. From there,
the c++ binary and the loop controller communicate directly via TCP.

This setup allows us to relaunch the c++ binary process as needed under the hood of a single
ratings server process. This is useful because it allows us to launch each matchup requested by
the loop controller as a separate c++ binary process.
"""
import argparse

from alphazero.logic.build_params import BuildParams
from alphazero.servers.gaming.ratings_server import RatingsServer, RatingsServerParams
from util.logging_util import LoggingParams
from util.py_util import CustomHelpFormatter


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    RatingsServerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser, add_ffi_lib_path_option=False)

    return parser.parse_args()


def main():
    args = load_args()
    params = RatingsServerParams.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    server = RatingsServer(params, logging_params, build_params)
    server.run()


if __name__ == '__main__':
    main()
