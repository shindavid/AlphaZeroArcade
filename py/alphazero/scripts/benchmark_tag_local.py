#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from util import subprocess_util
from util.py_util import CustomHelpFormatter

import argparse
import json
import os
import signal

default_target_elo_gap = 100

def save_default_benchmark(game: str, benchmark_tag: str):
    """
    Save the default benchmark tag for a given game to a JSON file.

    This will create or overwrite the file:
        /workspace/output/{game}/benchmark_info.json
    """

    output_dir = f"/workspace/output/{game}"
    os.makedirs(output_dir, exist_ok=True)

    benchmark_info = {
        "benchmark_tag": benchmark_tag
    }

    file_path = os.path.join(output_dir, "benchmark_info.json")

    with open(file_path, 'w') as f:
        json.dump(benchmark_info, f, indent=4)

    print(f"Benchmark tag '{benchmark_tag}' saved to {file_path}")


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    parser.add_argument('--target-elo-gap', type=int, default=default_target_elo_gap,
                        help='Target ELO gap for benchmarking (default: 100)')
    args = parser.parse_args()
    run_params = RunParams.create(args)

    cmd = ['./py/alphazero/scripts/run_local.py',
           '--game', run_params.game,
           '--tag', run_params.tag,
           '--task-mode',
           '--run-benchmark-server',
           '--target-elo-gap', str(args.target_elo_gap),
           '--benchmarking-mode', str]

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess_util.Popen(cmd, stderr=None, text=True, bufsize=1,
                                    preexec_fn=os.setsid)
    save_default_benchmark(run_params.game, run_params.tag)


if __name__ == "__main__":
    main()
