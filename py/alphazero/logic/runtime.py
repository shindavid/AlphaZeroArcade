from alphazero.logic.run_params import RunParams

import atexit
import json
import os
import sys


RUNTIME_DIR = '/workspace/repo/.runtime'
LOCK_FILE = os.path.join(RUNTIME_DIR, 'lock')
FREEZE_FILE = os.path.join(RUNTIME_DIR, 'freeze.json')


def acquire_lock():
    os.makedirs(RUNTIME_DIR, exist_ok=True)

    if os.path.exists(LOCK_FILE):
        print("Another instance of run_local or benchmark_tag_local is already running. Exiting.")
        print(f"To force this instance to run, remove the lock file: {LOCK_FILE}")
        sys.exit(1)

    with open(LOCK_FILE, 'w') as f:
        f.write('locked')

    atexit.register(release_lock)


def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        print("Lock released.")


def load_freeze_data():
    if not os.path.exists(FREEZE_FILE):
        return {}
    with open(FREEZE_FILE, 'r') as f:
        return json.load(f)


def save_freeze_data(data):
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(FREEZE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def freeze_tag(run_params: RunParams):
    data = load_freeze_data()
    game = run_params.game
    tag = run_params.tag

    if game not in data:
        data[game] = []

    if tag in data[game]:
        print(f"Tag: '{tag}' for game: '{game}' is already frozen.")
        return

    data[game].append(tag)
    save_freeze_data(data)
    print(f"Tag: '{tag}' for game: '{game}' has been frozen.")


def is_frozen(run_params: RunParams) -> bool:
    data = load_freeze_data()

    game = run_params.game
    tag = run_params.tag

    if game not in data:
        return False

    return tag in data[game]
