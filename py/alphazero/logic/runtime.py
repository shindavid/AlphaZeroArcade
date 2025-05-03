from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

import json
import logging
import os
from typing import Any, Callable


logger = logging.getLogger(__name__)


def acquire_lock(run_params: RunParams, register_func: Callable[[Callable[[], None]], Any]) -> str:
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    lock_file = os.path.join(runtime_dir, 'lock')

    if os.path.exists(lock_file):
        raise RuntimeError(
            f"Another instance of run_local or benchmark_tag_local is already running.\n"
            f"Exiting. To force this instance to run, remove the lock file: {lock_file}")

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

    logger.info(f"Benchmark tag '{benchmark_tag}' saved to {file_path}")
