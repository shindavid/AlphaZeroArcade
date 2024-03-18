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

from alphazero.servers.self_play.self_play_server import SelfPlayServer, SelfPlayServerParams
from util.logging_util import LoggingParams


def load_args():
    parser = argparse.ArgumentParser()

    SelfPlayServerParams.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    params = SelfPlayServerParams.create(args)
    logging_params = LoggingParams.create(args)

    server = SelfPlayServer(params, logging_params)
    server.run()


if __name__ == '__main__':
    main()
