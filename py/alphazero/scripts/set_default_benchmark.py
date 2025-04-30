#!/usr/bin/env python3
from alphazero.scripts.benchmark_tag_local import DEFAULT_TARGET_ELO_GAP
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.run_params import RunParams
from alphazero.logic.runtime import freeze_tag
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter

import numpy as np

import argparse
import json
import os


def save_default_benchmark(run_params: RunParams):
    """
    Save the default benchmark tag for a given game to a JSON file.

    This will create or overwrite the file:
        /workspace/output/{game}/benchmark_info.json
    """
    game = run_params.game
    benchmark_tag = run_params.tag
    output_dir = f"/workspace/output/{game}"
    os.makedirs(output_dir, exist_ok=True)

    benchmark_info = {
        "benchmark_tag": benchmark_tag
    }

    file_path = os.path.join(output_dir, "benchmark_info.json")

    with open(file_path, 'w') as f:
        json.dump(benchmark_info, f, indent=4)

    print(f"Benchmark tag '{benchmark_tag}' saved to {file_path}")


def is_valid_benchmark(run_params: RunParams):
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    benchmarker = Benchmarker(organizer)
    benchmark_rating_data: BenchmarkRatingData = benchmarker.read_ratings_from_db()
    latest_gen_in_model = organizer.get_latest_model_generation()
    latest_evaluated_gen = max(benchmark_rating_data.iagents, key=lambda ia: ia.agent.gen).agent.gen
    if latest_gen_in_model != latest_evaluated_gen:
        print(f"Benchmark Invalid: Latest evaluated gen {latest_evaluated_gen} does not match latest gen in model {latest_gen_in_model}")
        return False
    committee = benchmark_rating_data.committee
    matches = benchmarker.get_next_matches(n_iters=100,
                                           target_elo_gap=DEFAULT_TARGET_ELO_GAP,
                                           n_games=100,
                                           excluded_indices=~committee)
    if len(matches) > 0:
        print(f"Benchmark Invalid: There are more matches to be played.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = parser.parse_args()
    run_params = RunParams.create(args)
    if is_valid_benchmark(run_params):
        print("Benchmark is valid.")
        save_default_benchmark(run_params)
        freeze_tag(run_params)
    else:
        print("Benchmark is invalid. Please make sure benchmarking is complete and valid.")


if __name__ == "__main__":
    main()
