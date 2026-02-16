#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.params import LoopControllerParams
from util import subprocess_util
from util.py_util import CustomHelpFormatter

import argparse


default_target_rating_rate = LoopControllerParams().target_rating_rate


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--benchmark-tag', type=str, default=None,
                        help='Tag for the benchmark to be evaluated against')
    parser.add_argument('--target-rating-rate', type=float, default=default_target_rating_rate,
                        help='Target rating rate for the benchmark (default: 0.5)')
    args = parser.parse_args()
    run_params = RunParams.create(args)

    cmd = ['./py/alphazero/scripts/run_local.py',
           '--game', run_params.game,
           '--tag', run_params.tag,
           '--task-mode',
           '--run-eval-server']

    if args.benchmark_tag:
        cmd.extend(['--benchmark-tag', args.benchmark_tag])

    if args.target_rating_rate is not None and args.target_rating_rate != default_target_rating_rate:
        cmd.extend(['--target-rating-rate', str(args.target_rating_rate)])

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess_util.Popen(cmd, stdout=None, stderr=None)
    subprocess_util.wait_for(process, expected_return_code=None)


if __name__ == "__main__":
    main()
