#!/usr/bin/env python3
"""
Designate an *existing* AlphaZero run as a **benchmark**-run directory that can later be
used to evaluate fresh runs.

The script:
1.  Re-plays each selected agent *against itself* to stabilise ratings.
2.  Chooses a **benchmark committee**—agents whose Elo ratings differ by at least
    ``target_elo_gap``—to give a well-spaced difficulty ladder.
3.  Copies only the data needed for evaluation into a compact, timestamped folder.

/workspace/mount/benchmark_data/{game}/{tag}/{timestamp}_UTC

This script will create a folder with timestamp 1-01-01_00-00-00.000000_UTC because it will not be
uploaded. The uploaded folders will be named with the timestamp of the time of uploading.

1-01-01_00-00-00.000000_UTC/
    └── models/                 # Only the committee models
        ├── gen-7.pt
        ├── gen-14.pt
        └── …
    ├── binary
    ├── ratings.json
    ├── self_play.db
    └── training.db

models/ only contains the benchmark committee's model files. The benchmark committee is selected
with elo gaps greater than the configured target_elo_gap parameter to ensure proper spacing between.

json example
{
  "cmd_used": "./py/alphazero/scripts/benchmark_tag_local.py -g hex -t first-run",
  "MCTSAgent-gen-7": {
    "iagent": {
      "agent": {
        "type": "MCTS",
        "data": { "gen": 7, "n_iters": 100, "set_temp_zero": true,
                  "tag": "first-run", "binary": null, "model": null }
      },
      "index": 12,
      "roles": "benchmark",
      "db_id": 13
    },
    "rating": -26.91803106954515
  },
  "MCTSAgent-gen-14": {...}
}

How the folder is used

When loop_controller evaluates a new run, it expands the compact benchmark folder
into a pseudo-run directory, which contains a subset of folders of a regular run folder:
/workspace/mount/output/{game}/{tag}.benchmark
    ├── bin/hex
    ├── databases/
    │   ├── benchmark.db
    │   ├── self_play.db
    │   └── training.db
    ├── misc/version_file
    └── models/
        ├── gen-7.pt
        ├── gen-14.pt
        └── ...
"""
from alphazero.logic.benchmark_record import save_benchmark_data, UTC_FORMAT, BenchmarkRecord, \
    BenchmarkOption
from alphazero.logic.build_params import BuildParams
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from shared.rating_params import RatingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter

import argparse
from datetime import datetime, timezone
import logging
import os
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_rating_params = None if game_spec is None else game_spec.rating_params
    RatingParams.add_args(parser, defaults=default_rating_params)
    BuildParams.add_args(parser, loop_controller=True)
    LoggingParams.add_args(parser)
    return parser.parse_args()


def get_benchmark_cmd(run_params: RunParams, build_params: BuildParams, rating_params: RatingParams,
                      logging_params: LoggingParams):
    cmd = ['./py/alphazero/scripts/run_local.py',
           '--task-mode',
           '--run-benchmark-server']

    if rating_params.target_elo_gap is not None:
        cmd.extend(['--target-elo-gap', str(rating_params.target_elo_gap)])
    else:
        cmd.extend(['--target-elo-gap', str(rating_params.default_target_elo_gap.benchmark)])

    logging_params.add_to_cmd(cmd)
    run_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, loop_controller=True)
    rating_params.add_to_cmd(cmd, loop_controller=True, server=True)
    return cmd


def get_eval_cmd(run_params: RunParams, build_params: BuildParams, rating_params: RatingParams,
                 logging_params: LoggingParams, benchmark_tag: str):
    cmd = ['./py/alphazero/scripts/run_local.py',
           '--task-mode',
           '--run-eval-server']
    assert benchmark_tag is not None, \
        "Benchmark tag should not be None after running benchmark server."
    cmd.extend(['--benchmark-tag', benchmark_tag])

    logging_params.add_to_cmd(cmd)
    run_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, loop_controller=True)
    rating_params.add_to_cmd(cmd, loop_controller=True, server=True)
    return cmd


def main():
    args = load_args()
    run_params = RunParams.create(args)
    organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
    if not os.path.exists(organizer.base_dir):
        raise Exception(f"Run {organizer.base_dir} does not exist.")

    organizer.assert_unlocked()

    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)
    rating_params = RatingParams.create(args)
    configure_logger(params=logging_params, prefix='[benchmark_tag_local]')

    benchmark_cmd = get_benchmark_cmd(run_params, build_params, rating_params, logging_params)
    logger.info(f"Running command: {' '.join(benchmark_cmd)}")

    try:
        subprocess.run(benchmark_cmd, text=True, check=True)
    except:
        logger.error(f"Command: {benchmark_cmd} failed.")
        return

    organizer.freeze_tag()
    zero_time = datetime(1, 1, 1, tzinfo=timezone.utc)
    utc_key = zero_time.strftime(UTC_FORMAT)
    record = BenchmarkRecord(utc_key=utc_key, tag=run_params.tag, game=run_params.game)
    save_benchmark_data(organizer, record)

    eval_cmd = get_eval_cmd(run_params, build_params, rating_params, logging_params, organizer.tag)
    logger.info(f"Running command: {' '.join(eval_cmd)}")
    try:
        subprocess.run(eval_cmd, text=True, check=True)
    except:
        logger.error(f"Command: {eval_cmd} failed.")
        return
    logger.info("Benchmark evaluation completed successfully.")


if __name__ == "__main__":
    main()
