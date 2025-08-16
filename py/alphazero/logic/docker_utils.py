"""
Contains some utility functions to inspect a Docker image, intended to be called from within a
Docker container.
"""
try:
    from setup_common import MINIMUM_REQUIRED_IMAGE_VERSION, is_version_ok
except ImportError:
    import sys
    sys.path.append('/workspace/repo')
    from setup_common import MINIMUM_REQUIRED_IMAGE_VERSION, is_version_ok

from dataclasses import dataclass
import os
import sys
from typing import List


@dataclass
class DockerParams:
    skip_image_version_check: bool

    @staticmethod
    def create(args) -> 'DockerParams':
        return DockerParams(
            skip_image_version_check=bool(args.skip_image_version_check),
        )

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('Docker options')
        group.add_argument('--skip-image-version-check', action='store_true',
                           help='skip docker image version check')

    def add_to_cmd(self, cmd: List[str]):
        if self.skip_image_version_check:
            cmd.append('--skip-image-version-check')


def validate_docker_image():
    """
    Validate that the Docker image is up-to-date. Calls sys.exit(1) if the image is out of date.
    """
    version_str = os.getenv('DOCKER_IMAGE_VERSION', None)
    if not is_version_ok(version_str):
        if version_str is None:
            print('Your docker image appears out of date.')
        else:
            print('Your docker image version (%s) is out of date (required: %s).' %
                  (version_str, MINIMUM_REQUIRED_IMAGE_VERSION))
        print('')
        print('Please do the following:')
        print('  1. Exit all ./run_docker.py instances')
        print('  2. Refresh your docker image by running ./pull_docker_image.py')
        print('  3. Relaunch ./run_docker.py')
        print('')
        print('Alternatively, rerun this command with the --skip-image-version-check flag.')
        print('')
        sys.exit(1)
