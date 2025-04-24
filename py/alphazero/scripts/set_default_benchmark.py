#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from util.py_util import CustomHelpFormatter

import argparse
import json
import os


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
    args = parser.parse_args()
    run_params = RunParams.create(args)
    save_default_benchmark(run_params.game, run_params.tag)


if __name__ == "__main__":
    main()