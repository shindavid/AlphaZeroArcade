#!/usr/bin/env python3
from alphazero.logic.agent_types import IndexedAgent
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.benchmark_record import UTC_FORMAT
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import BenchmarkRecord, Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.aws_util import BUCKET
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter, tar_and_remotely_copy

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import shlex
import shutil
import sys


logger = logging.getLogger(__name__)


def save_benchmark_data(organizer: DirectoryOrganizer, record: BenchmarkRecord):
    benchmarker = Benchmarker(organizer)
    path = record.data_folder_path()
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
    logger.info(f"Created benchmark data folder {path}")


def save_benchmark_record(record: BenchmarkRecord):
    benchmark_info = record.to_dict()
    benchmark_record_file = Workspace.benchmark_record_file(record.game)

    os.makedirs(os.path.dirname(benchmark_record_file), exist_ok=True)
    with open(benchmark_record_file, 'w') as f:
        json.dump(benchmark_info, f, indent=4)

    logger.info(f"Benchmark tag '{record.tag}' saved to {benchmark_record_file}")


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    LoggingParams.add_args(parser)
    parser.add_argument('--skip-set-as-default', action='store_true',
                        help='Skip setting the benchmark as default.')
    return parser.parse_args()


def main():
    args = load_args()
    logging_params = LoggingParams.create(args)
    run_params = RunParams.create(args)
    configure_logger(params=logging_params, prefix='[upload_benchmark_to_s3]')
    utc_key = datetime.now(timezone.utc).strftime(UTC_FORMAT)
    record = BenchmarkRecord(utc_key=utc_key, tag=run_params.tag, game=run_params.game)

    organizer = DirectoryOrganizer(RunParams(args.game, args.tag), base_dir_root=Workspace)
    if not os.path.isdir(organizer.base_dir):
        raise FileNotFoundError(f"dir {organizer.base_dir} does not exist.")
    save_benchmark_data(organizer, record)
    if not args.skip_set_as_default:
        save_benchmark_record(record)
    tar_file = f"{record.data_folder_path()}.tar"
    tar_and_remotely_copy(record.data_folder_path(), tar_file)
    BUCKET.upload_file_to_s3(tar_file, record.key())


if __name__ == '__main__':
    main()
