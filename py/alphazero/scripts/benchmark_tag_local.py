#!/usr/bin/env python3
from alphazero.logic.agent_types import IndexedAgent
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.build_params import BuildParams
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.scripts.run_local import get_benchmark_tag
from alphazero.servers.loop_control.base_dir import Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from shared.rating_params import RatingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter

import argparse
from datetime import datetime, timezone
import logging
import json
import os
import shlex
import shutil
import subprocess
import sys
from typing import Optional


logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_rating_params = None if game_spec is None else game_spec.rating_params
    RatingParams.add_args(parser, defaults=default_rating_params)
    BuildParams.add_args(parser, loop_controller=True)
    LoggingParams.add_args(parser)
    parser.add_argument('--skip-set-as-default', action='store_true',
                        help='Skip setting the benchmark as default.')
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
                 logging_params: LoggingParams):
    benchmark_tag = get_benchmark_tag(run_params.game)
    cmd = ['./py/alphazero/scripts/run_local.py',
           '--task-mode',
           '--run-eval-server']
    assert benchmark_tag is not None, "Benchmark tag should not be None after running benchmark server."
    cmd.extend(['--benchmark-tag', benchmark_tag])

    logging_params.add_to_cmd(cmd)
    run_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, loop_controller=True)
    rating_params.add_to_cmd(cmd, loop_controller=True, server=True)
    return cmd

def save_benchmark_data(organizer: DirectoryOrganizer, utc_key: str):
    benchmarker = Benchmarker(organizer)
    path = os.path.join(Workspace.benchmark_data_dir, organizer.game, utc_key, organizer.tag)
    model_path = os.path.join(path, 'models')
    os.makedirs(model_path, exist_ok=True)
    file = os.path.join(path, 'ratings.json')
    rating_data: BenchmarkRatingData = benchmarker.read_ratings_from_db()

    ix = 0
    db_id = 1
    indexed_agents = []
    ratings = []
    for i in rating_data.committee:
        ia: IndexedAgent = rating_data.iagents[i]
        ia.index = ix
        ia.db_id = db_id
        ix += 1
        db_id += 1
        indexed_agents.append(ia)
        ratings.append(rating_data.ratings[i])
        gen = ia.agent.gen
        if gen == 0:
            continue
        src = organizer.get_model_filename(gen)
        shutil.copyfile(src, os.path.join(model_path, f'gen-{gen}.pt'))

    indexed_agents, ratings = zip(*sorted(zip(indexed_agents, ratings), key=lambda x: x[1]))
    cmd = shlex.join(sys.argv)
    RatingDB.save_ratings_to_json(indexed_agents, ratings, file, cmd)
    shutil.copyfile(organizer.binary_filename, os.path.join(path, 'binary'))
    shutil.copyfile(organizer.self_play_db_filename, os.path.join(path, 'self_play.db'))
    shutil.copyfile(organizer.training_db_filename, os.path.join(path, 'training.db'))


def save_benchmark_info(game: str, tag: str, utc_key: str):
    benchmark_info = {
        "utc_key": utc_key,
        "tag": tag
    }
    benchmark_info_file = Workspace.benchmark_info_file(game)
    with open(benchmark_info_file, 'w') as f:
        json.dump(benchmark_info, f, indent=4)

    logger.info(f"Benchmark tag '{tag}' saved to {benchmark_info_file}")


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
    utc_key = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    if not args.skip_set_as_default:
        save_benchmark_info(run_params.game, run_params.tag, utc_key)

    save_benchmark_data(organizer, utc_key)

    eval_cmd = get_eval_cmd(run_params, build_params, rating_params, logging_params)
    logger.info(f"Running command: {' '.join(eval_cmd)}")
    try:
        subprocess.run(eval_cmd, text=True, check=True)
    except:
        logger.error(f"Command: {eval_cmd} failed.")
        return
    logger.info("Benchmark evaluation completed successfully.")


if __name__ == "__main__":
    main()
