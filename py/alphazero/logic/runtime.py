from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

import atexit
import os
import sys


def acquire_lock(run_params: RunParams):
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    lock_file = os.path.join(runtime_dir, 'lock')

    if os.path.exists(lock_file):
        print("Another instance of run_local or benchmark_tag_local is already running. Exiting.")
        print(f"To force this instance to run, remove the lock file: {lock_file}")
        sys.exit(1)

    with open(lock_file, 'w') as f:
        f.write('locked')

    atexit.register(release_lock, lock_file)


def release_lock(lock_file: str):
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print("Lock released.")

def freeze_tag(run_params: RunParams):
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    freeze_file = os.path.join(runtime_dir, 'freeze')
    with open(freeze_file, 'w') as f:
        f.write('frozen')
    print(f"Froze run {run_params.game}: {run_params.tag}.")


def is_frozen(run_params: RunParams) -> bool:
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    runtime_dir = organizer.runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    freeze_file = os.path.join(runtime_dir, 'freeze')
    return os.path.exists(freeze_file)
