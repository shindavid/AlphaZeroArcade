#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from setup_common import LATEST_DOCKER_HUB_IMAGE, update_env_json

import argparse
from pathlib import Path
import os
import pty
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--docker-hub-image', default=LATEST_DOCKER_HUB_IMAGE,
                        help='Docker Hub image name (default: %(default)s)')
    return parser.parse_args()


def docker_pull(image):
    print(f'Pulling {image}...')
    cmd = ["docker", "pull", image]

    # Open a pseudo-terminal pair
    master_fd, slave_fd = pty.openpty()

    # Start the subprocess with the slave end as its stdout and stderr.
    process = subprocess.Popen(cmd, stdout=slave_fd, stderr=slave_fd, universal_newlines=True)
    os.close(slave_fd)  # close slave fd in the parent process

    # Read from the master end and write to sys.stdout.
    try:
        while True:
            output = os.read(master_fd, 1024)
            if not output:
                break
            sys.stdout.write(output.decode())
            sys.stdout.flush()
    except OSError:
        pass
    process.wait()

    update_env_json({'DOCKER_IMAGE': image})

    if process.returncode == 0:
        print('✅ Docker pull successful.')
    else:
        # Handle unexpected output or errors
        print('❗ Unexpected output from docker pull.')
        # Optionally, you can raise an exception or handle it as needed
        raise RuntimeError("Unexpected output from docker pull.")


def blow_away_target_dir():
    repo_root = Path(__file__).parent.resolve()
    target_dir = repo_root / 'target'
    print(f'Blowing away {target_dir}...')
    os.system(f'rm -rf {target_dir}')


def main():
    args = get_args()
    docker_pull(args.docker_hub_image)


if __name__ == '__main__':
    main()
