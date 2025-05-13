#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from games.game_spec import GameSpec
from shared.rating_params import RatingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter

import argparse
import logging
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)
DEFAULT_TARGET_ELO_GAP = 100


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_rating_params = None if game_spec is None else game_spec.rating_params
    RatingParams.add_args(parser, defaults=default_rating_params)
    LoggingParams.add_args(parser)
    parser.add_argument('--skip-set-as-default', action='store_true',
                        help='Skip setting the benchmark as default.')
    return parser.parse_args()

def main():
    args = load_args()
    run_params = RunParams.create(args)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    organizer.assert_unlocked()

    logging_params = LoggingParams.create(args)
    configure_logger(params=logging_params, prefix='[benchmark_tag_local]')

    cmd = ['./py/alphazero/scripts/run_local.py',
           '--task-mode',
           '--run-benchmark-server',
           '--target-elo-gap', str(args.benchmark_target_elo_gap),]
    logging_params.add_to_cmd(cmd)
    run_params.add_to_cmd(cmd)

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, text=True, check=True)
    except:
        logger.error(f"Command: {cmd} failed.")
        return

    organizer.freeze_tag()
    if not args.skip_set_as_default:
        organizer.save_default_benchmark()


if __name__ == "__main__":
    main()
