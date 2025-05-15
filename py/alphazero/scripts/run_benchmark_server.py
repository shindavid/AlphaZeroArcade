#!/usr/bin/env python3

"""
Entry point for launching the benchmark server.

This script connects to the loop controller and participates in the distributed benchmarking
process. It listens for match requests, runs benchmark matches between committee members and
new agents, and reports match results back to the controller.

* Relevant Loop Controller Parameters

1. `--benchmark_until_gen_gap`:
   The minimum number of generations between the latest evaluated generation and the latest
   trained generation before benchmarking is prioritized.

2. `--n_games_per_benchmark`:
   The number of games to play in each benchmark match between two agents.

3. `--target_elo_gap`:
   The maximum allowable Elo gap between two agents in a match (used to determine who needs
   to play whom for accurate rating computation), and also the **minimum** required Elo gap
   between committee members to ensure they represent a broad distribution of skill levels.
"""

from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.servers.gaming.benchmark_server import BenchmarkServer, BenchmarkServerParams
from shared.rating_params import RatingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo

import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    BenchmarkServerParams.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser)
    RatingParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    params = BenchmarkServerParams.create(args)
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

    server = BenchmarkServer(params, logging_params, build_params, rating_params)
    server.run()


if __name__ == '__main__':
    main()
