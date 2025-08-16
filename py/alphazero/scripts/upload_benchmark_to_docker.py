#!/usr/bin/env python3
"""
This script packages and uploads a benchmark to Docker Hub.

Workflow:
1. Validate that the benchmark folder for the given game and tag exists.
2. Archive (tar) the benchmark folder into a tar file.
3. Build a minimal Docker image that contains only this tar file at /payload/artifact.tar, with
   metadata labels such as version, game, tag, utc_key, and sha256.
4. Push the resulting image to Docker Hub under tag {version}.{game}.{tag}.{utc}.
5. Save the benchmark record in /workspace/repo/benchmark_records/{game}.json.

Notes:
  - Uploading the same tar file multiple times does not duplicate storage in Docker Hub;
    identical content will be stored once and referenced by multiple tags if necessary.
"""

from alphazero.logic.benchmark_record import BenchmarkRecord, build_single_file_docker_image, \
        save_benchmark_record, upload_image, UTC_FORMAT
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Benchmark
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter, tar_and_remotely_copy

import argparse
from datetime import datetime, timezone
import logging
import os


logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    LoggingParams.add_args(parser)
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

    tar_file = Benchmark.tar_path(record.game, record.tag, utc_key=record.utc_key)
    os.makedirs(os.path.dirname(tar_file), exist_ok=True)
    tar_and_remotely_copy(folder, tar_file)
    build_single_file_docker_image(tar_file, record)
    upload_image(record)
    save_benchmark_record(record)


if __name__ == '__main__':
    main()
