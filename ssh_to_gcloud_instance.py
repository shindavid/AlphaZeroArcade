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
import os
from pathlib import Path
import shlex
import subprocess
import time


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

    # chdir to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # First, do an initial ssh to the instance to do the following:
    #
    # 1. Run "sudo usermod -aG docker $USER"
    # 2. Copy gcloud/default_bashrc to ~/.bashrc
    #
    # Ideally, these are executed only once when the user is created on the instance. But I couldn't
    # get that to work properly, so I'm doing it here.

    default_bashrc_str = Path('gcloud/default_bashrc').read_text().strip()

    cmd = [
        "gcloud", "compute", "ssh", f"{params.username}@{params.instance}",
        f"--zone={params.zone}",
        "--command",
        f"sudo usermod -aG docker $USER; echo {shlex.quote(default_bashrc_str)} > ~/.bashrc"
    ]

    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    print(f"Running: {cmd_str}")

    # Retry a few times in case the instance is not ready yet.
    retry_count = 5
    sleep_time = 5
    for i in range(retry_count):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            break
        except subprocess.CalledProcessError as e:
            if i == retry_count - 1:
                print(f'Giving up after {retry_count} retries!')
                raise e
            print("Cmd failed. This might be because he instance is still starting up.")
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    cmd = [
        'gcloud', 'compute', 'ssh', f'{params.username}@{params.instance}',
        f'--zone={params.zone}',
    ]

    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    print(f'Running: {cmd_str}')
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
