from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

import logging
import os
import sys
from typing import Any, Callable


logger = logging.getLogger(__name__)


def acquire_lock(run_params: RunParams, register_func: Callable[[Callable[[], None]], Any]) -> str:
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    lock_file = os.path.join(runtime_dir, 'lock')

    if os.path.exists(lock_file):
        logger.info("Another instance of run_local or benchmark_tag_local is already running. Exiting.")
        logger.info(f"To force this instance to run, remove the lock file: {lock_file}")
        sys.exit(1)

    with open(lock_file, 'w') as f:
        f.write('locked')
    register_func(lambda: release_lock(lock_file))


def release_lock(lock_file: str):
    if os.path.exists(lock_file):
        os.remove(lock_file)
        logger.info("Lock released.")


def freeze_tag(run_params: RunParams):
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    freeze_file = os.path.join(runtime_dir, 'freeze')
    with open(freeze_file, 'w') as f:
        f.write('frozen')
    logger.info(f"Froze run {run_params.game}: {run_params.tag}.")


def is_frozen(run_params: RunParams) -> bool:
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    freeze_file = os.path.join(runtime_dir, 'freeze')
    return os.path.exists(freeze_file)
