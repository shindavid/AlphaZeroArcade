#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from setup_common import LATEST_DOCKER_HUB_IMAGE, update_env_json

import argparse
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


def main():
    args = get_args()
    docker_pull(args.docker_hub_image)


if __name__ == '__main__':
    main()
