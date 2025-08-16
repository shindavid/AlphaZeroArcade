#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

# This script builds a local Docker image based on the contents of the docker-setup/ directory.
# Currently, this is only intended to be run by user dshin.

from setup_common import in_docker_container, LOCAL_DOCKER_IMAGE

import argparse
import os
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--local-docker-image', default=LOCAL_DOCKER_IMAGE,
                        help='Local docker image name (default: %(default)s)')
    return parser.parse_args()


def docker_build(image):
    os.chdir(os.path.dirname(__file__))
    print(f'Building docker image {image}...')

    cmd = ['docker', 'build', '-t', image, 'docker-setup/']
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Failed to build docker image {image}!")
        return 1

    print(f"✅ Successfully built docker image {image}!")
    return 0


def main():
    args = get_args()
    sys.exit(docker_build(args.local_docker_image))


if __name__ == '__main__':
    assert not in_docker_container(), "This script should not be run inside a0a docker container!"
    main()
