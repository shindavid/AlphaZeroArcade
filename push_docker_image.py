#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

# This script pushes a local docker image to Docker Hub. It assumes that you are currently logged
# in as the username in the Docker Hub image name. Currently, this is only intended to be run by
# user dshin.

from setup_common import DOCKER_HUB_IMAGE, LOCAL_DOCKER_IMAGE

import argparse
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--local-docker-image', default=LOCAL_DOCKER_IMAGE,
                        help='Local docker image name (default: %(default)s)')
    parser.add_argument("-i", '--docker-hub-image', default=DOCKER_HUB_IMAGE,
                        help='Docker Hub image name (default: %(default)s)')
    return parser.parse_args()


def get_image_id(local_image):
    result = subprocess.check_output(
        ["docker", "images", "-q", local_image],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()
    if not result:
        raise Exception(f"Image {local_image} not found.")
    print(f'Found image {local_image} with id {result}')
    return result


def docker_tag(image_id, remote_image):
    print(f'Tagging {image_id} as {remote_image}...')
    subprocess.run(['docker', 'tag', image_id, remote_image], check=True)


def docker_push(remote_image):
    print(f'Pushing {remote_image}...')
    subprocess.run(['docker', 'push', remote_image], check=True)


def main():
    args = get_args()
    image_id = get_image_id(args.local_docker_image)
    docker_tag(image_id, args.docker_hub_image)
    docker_push(args.docker_hub_image)
    print('✅ Successfully pushed docker image!')


if __name__ == '__main__':
    main()
