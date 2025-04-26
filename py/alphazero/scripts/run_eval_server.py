#!/usr/bin/env python3

"""
Entry point for launching the evaluation server.

This script connects to the loop controller and participates in the distributed evaluation process.
Evaluation is performed against a previously generated benchmark (produced by the benchmark server),
and the resulting Elo ratings are scaled relative to the agents in that benchmark.

The eval server listens for match requests, executes evaluation matches, and reports results back
to the loop controller. The eval manager (running in the controller) is responsible for selecting
which committee members each generation should play against.

### Relevant Loop Controller Parameters

1. `--target_rating_rate`
   The target percentage of available generations to be evaluated. If the actual percentage is below
   this value, evaluation is prioritized.

2. `--n_games_per_evaluation`
   The total number of games assigned to evaluate a single generation. This budget is distributed
   across committee members based on the expected win probability — assigning more games to
   agents that are likely to be close in skill. This focuses evaluation on the most informative matches.

3. `--eval_error_threshold`
   A threshold on the error (uncertainty) in Elo estimation. If the estimated Elo changes significantly
   after additional games, it means the initial assignment may have been suboptimal. In this case,
   match distribution is adjusted to better allocate remaining games.

4. `--agent_n_iters`
   The number of MCTS iterations used by the agent being evaluated. Committee agents use a fixed
   iteration count based on their benchmark definition; this parameter only affects the evaluation target.

5. `--benchmark-tag`
   The tag identifying the benchmark database (`benchmark.db`) to use for evaluation.
"""

from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.servers.gaming.eval_server import EvalServer, EvalServerParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo

import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    EvalServerParams.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser)

    return parser.parse_args()


def main():
    args = load_args()
    params = EvalServerParams.create(args)
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

    server = EvalServer(params, logging_params, build_params)
    server.run()


if __name__ == '__main__':
    main()
