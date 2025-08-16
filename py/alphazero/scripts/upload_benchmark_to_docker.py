#!/usr/bin/env python3
from alphazero.logic.benchmark_record import BenchmarkRecord, build_one_file_docker_image, UTC_FORMAT
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Benchmark, Workspace
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter, tar_and_remotely_copy

import argparse
from datetime import datetime, timezone
import json
import logging
import os


logger = logging.getLogger(__name__)


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

    folder = Benchmark.path(record.game, record.tag)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"dir {folder} does not exist.")

    if not args.skip_set_as_default:
        save_benchmark_record(record)

    tar_file = Benchmark.tar_path(record.game, record.tag, utc_key=record.utc_key)
    os.makedirs(os.path.dirname(tar_file), exist_ok=True)
    tar_and_remotely_copy(folder, tar_file)
    img = build_one_file_docker_image(tar_file, record)


if __name__ == '__main__':
    main()
