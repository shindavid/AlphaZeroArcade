#!/usr/bin/env python3

"""
Launches a GCP instance.

Many of the steps of this script invoke the gcloud CLI directly, rather than using the python client
library. This is because the python client library does not provide as much detail in the error
messages, which makes it harder to debug issues. Besides, the gcloud CLI is identical across
platforms (such as Windows, Mac, and Linux), so doing it this way is more portable. Finally, the
CLI is much simpler and easier to use than the python client library.
"""

from gcloud_common import Defaults, get_gcloud_zone
from setup_common import get_env_json, update_env_json

import argparse
from dataclasses import dataclass, fields
import shlex
import subprocess
from typing import Optional


@dataclass
class Params:
    image_name: str = ''  # default: latest from image family
    image_family: str = Defaults.image_family

    name: str = 'compute-instance'
    zone: str = get_gcloud_zone()
    machine_type: str = Defaults.machine_type
    ssd_disk_count: int = 1
    gpu_type: Optional[str] = Defaults.gpu_type
    gpu_count: int = 1

    disk_name: str = get_env_json().get('GCP_PERSISTENT_DISK', None)
    disk_mode: str = 'rw'

    preemptible: bool = False

    @staticmethod
    def create(args) -> 'Params':
        kwargs = {f.name: getattr(args, f.name) for f in fields(Params)}
        params = Params(**kwargs)
        if None in (params.disk_name, params.zone):
            raise ValueError('Please run gcp_setup_wizard.py')
        return params

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('launch_gcloud_instance.py options')

        defaults = Params()
        group.add_argument('-f', '--image-family', default=defaults.image_family,
                           help='Family of the image to use (default: %(default)s)')
        group.add_argument('-i', '--image-name', default=defaults.image_name,
                           help='Name of the image to use (if specified, overrides -f/--image-family)')
        group.add_argument('-n', '--name', default=defaults.name,
                           help='Name of the instance to create (default: %(default)s)')
        group.add_argument('-z', '--zone', default=defaults.zone,
                           help='Zone to create the instance in (default: %(default)s)')
        group.add_argument('-m', '--machine-type', default=defaults.machine_type,
                           help='Machine type to create the instance with (default: %(default)s)')
        group.add_argument('-g', '--gpu-type', default=defaults.gpu_type,
                           help='GPU type to create the instance with (default: %(default)s). '
                                'Empty-string means omit gpu.')
        group.add_argument('-c', '--gpu-count', default=defaults.gpu_count, type=int,
                           help='Number of GPUs to create the instance with (default: %(default)s).'
                                ' Only used if -g/--gpu-type is not empty-string.')
        group.add_argument('-s', '--ssd-disk-count', default=defaults.ssd_disk_count, type=int,
                           help='Number of SSD disks to create the instance with (default: %(default)s)')
        group.add_argument('-d', '--disk-name', default=defaults.disk_name,
                           help='Name of the disk to attach (default: %(default)s)')
        group.add_argument('-D', '--disk-mode', default=defaults.disk_mode,
                           help='Disk mode to attach (default: %(default)s)')
        group.add_argument('-p', '--preemptible', action='store_true',
                           help='Create a preemptible instance')


def load_args() -> Params:
    parser = argparse.ArgumentParser()
    Params.add_args(parser)
    return Params.create(parser.parse_args())


def main():
    params = load_args()

    cmd = [
        'gcloud', 'compute', 'instances', 'create', params.name,
        f'--zone={params.zone}',
        f'--machine-type={params.machine_type}',
        f'--disk=name={params.disk_name},mode={params.disk_mode},boot=no,auto-delete=no',
        '--maintenance-policy=TERMINATE',
        '--metadata-from-file=startup-script=gcloud/instance_startup.sh',
        f'--image-project={Defaults.a0a_project}',
    ]

    for _ in range(params.ssd_disk_count):
        cmd.append('--local-ssd=interface=nvme')

    if params.gpu_type:
        cmd.append(f'--accelerator=type={params.gpu_type},count={params.gpu_count}')

    if params.preemptible:
        cmd.append('--preemptible')

    if params.image_name:
        cmd.append(f'--image={params.image_name}')
    else:
        cmd.append(f'--image-family={params.image_family}')

    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    print(f'About to run command: {cmd_str}')
    print('')
    print('TODO: provide an estimate of the hourly cost of this instance.')
    print('TODO: warn user to delete the instance when done.')
    print('')
    print('Press enter to continue...')
    input()
    print('Launching. This may take a few minutes...')
    subprocess.run(cmd, check=True)
    print('')
    print('✅ Successfully launched gcloud instance!')
    print('')
    print('To connect to this instance, please run:')
    print('')
    print('./ssh_to_gcloud_instance.py')
    print('')
    print('To monitor, please visit: https://console.cloud.google.com/compute/instances')
    print('')
    update_env_json({'GCP_INSTANCE': params.name})


if __name__ == "__main__":
    main()
