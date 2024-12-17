#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from setup_common import LATEST_DOCKER_HUB_IMAGE, update_env_json

import argparse
from pathlib import Path
import os
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--docker-hub-image', default=LATEST_DOCKER_HUB_IMAGE,
                        help='Docker Hub image name (default: %(default)s)')
    return parser.parse_args()


def docker_pull(image):
    print(f'Pulling {image}...')
    subprocess.run(['docker', 'pull', image], check=True)
    update_env_json({'DOCKER_IMAGE': image})
    print('✅ Successfully pulled docker image!')


def docker_pull(image):
    print(f'Pulling {image}...')

    # Run the docker pull command and capture the output
    result = subprocess.run(
        ['docker', 'pull', image],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = result.stdout
    print(output)  # Optionally print the output for debugging

    # Determine if the image was updated
    # Common indicators:
    # - "Status: Image is up to date" means no update
    # - "Downloaded newer image" or "Status: Downloaded newer image" means updated
    if "Status: Image is up to date" in output:
        print('✅ Image was already up-to-date.')
    elif any(phrase in output for phrase in ["Downloaded newer image", "Status: Downloaded newer image"]):
        print('✅ Successfully pulled a newer version of the image!')
        blow_away_target_dir()
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
