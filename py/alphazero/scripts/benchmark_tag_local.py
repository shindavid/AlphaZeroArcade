#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from util import subprocess_util
from util.py_util import CustomHelpFormatter

import argparse


DEFAULT_TARGET_ELO_GAP = 100


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--target-elo-gap', type=int, default=DEFAULT_TARGET_ELO_GAP,
                        help=f'Target ELO gap for benchmarking (default: {DEFAULT_TARGET_ELO_GAP})')
    args = parser.parse_args()
    run_params = RunParams.create(args)

    cmd = ['./py/alphazero/scripts/run_local.py',
           '--game', run_params.game,
           '--tag', run_params.tag,
           '--task-mode',
           '--run-benchmark-server',
           '--target-elo-gap', str(args.target_elo_gap)]

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess_util.Popen(cmd, stdout=None, stderr=None)
    subprocess_util.wait_for(process, expected_return_code=None)


if __name__ == "__main__":
    main()
