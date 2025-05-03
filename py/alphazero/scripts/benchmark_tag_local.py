#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from alphazero.logic.runtime import freeze_tag, save_default_benchmark
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter

import argparse
import logging
import os
import subprocess


logger = logging.getLogger(__name__)
DEFAULT_TARGET_ELO_GAP = 100


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    LoggingParams.add_args(parser)
    parser.add_argument('--target-elo-gap', type=int, default=DEFAULT_TARGET_ELO_GAP,
                        help=f'Target ELO gap for benchmarking (default: {DEFAULT_TARGET_ELO_GAP})')
    args = parser.parse_args()
    run_params = RunParams.create(args)
    logging_params = LoggingParams.create(args)
    configure_logger(params=logging_params, prefix='[benchmark_tag_local]')

    cmd = ['./py/alphazero/scripts/run_local.py',
           '--game', run_params.game,
           '--tag', run_params.tag,
           '--task-mode',
           '--run-benchmark-server',
           '--target-elo-gap', str(args.target_elo_gap)]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, text=True, check=True)
    except:
        logger.error(f"Command: {cmd} failed.")
        return

    save_default_benchmark(run_params)
    freeze_tag(run_params)

if __name__ == "__main__":
    main()
