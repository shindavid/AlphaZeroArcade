#!/usr/bin/env python3

"""
Connects to a GCP instance.

Many of the steps of this script invoke the gcloud CLI directly, rather than using the python client
library. This is because the python client library does not provide as much detail in the error
messages, which makes it harder to debug issues. Besides, the gcloud CLI is identical across
platforms (such as Windows, Mac, and Linux), so doing it this way is more portable. Finally, the
CLI is much simpler and easier to use than the python client library.
"""

from gcloud_common import Defaults, get_gcloud_zone
from setup_common import get_env_json

import argparse
from dataclasses import dataclass, fields
import subprocess


@dataclass
class Params:
    username: str = Defaults.username
    instance: str = get_env_json().get('GCP_INSTANCE', None)
    zone: str = get_gcloud_zone()

    @staticmethod
    def create(args) -> 'Params':
        kwargs = {f.name: getattr(args, f.name) for f in fields(Params)}
        params = Params(**kwargs)
        if params.zone is None:
            raise ValueError('Please run gcp_setup_wizard.py')
        if params.instance is None:
            raise ValueError('Please run launch_gcloud_instance.py')
        return params

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('launch_gcloud_instance.py options')

        defaults = Params()
        group.add_argument('-u', '--username', default=defaults.username,
                           help='Username (default: %(default)s)')
        group.add_argument('-i', '--instance', default=defaults.instance,
                           help='Name of instance (default: %(default)s)')
        group.add_argument('-z', '--zone', default=defaults.zone,
                           help='Zone to create the instance in (default: %(default)s)')


def load_args() -> Params:
    parser = argparse.ArgumentParser()
    Params.add_args(parser)
    return Params.create(parser.parse_args())


def main():
    params = load_args()

    # First cp gcloud/.bashrc to the instance
    cmd = [
        'gcloud', 'compute', 'scp', f'gcloud/default_bashrc',
        f'{params.username}@{params.instance}:~/.bashrc',
        f'--zone={params.zone}',
    ]
    cmd_str = ' '.join(cmd)
    print(f'Running: {cmd_str}')
    subprocess.run(cmd, check=True, capture_output=True)

    cmd = [
        'gcloud', 'compute', 'ssh', f'{params.username}@{params.instance}',
        f'--zone={params.zone}',
    ]

    cmd_str = ' '.join(cmd)
    print(f'Running: {cmd_str}')
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
