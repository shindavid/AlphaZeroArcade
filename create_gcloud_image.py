#!/usr/bin/env python3

"""
Creates a custom GCP image, to be used for instance creation. Only the GCP project owner (dshin)
needs to run this script. Other users, as long as they are part of the AlphaZeroArcade GCP project,
can launch instances from the custom image.

Many of the steps of this script invoke the gcloud CLI directly, rather than using the python client
library. This is because the python client library does not provide as much detail in the error
messages, which makes it harder to debug issues. Besides, the gcloud CLI is identical across
platforms (such as Windows, Mac, and Linux), so doing it this way is more portable. Finally, the
CLI is much simpler and easier to use than the python client library.
"""

from gcloud_common import Defaults, Help, get_gcloud_zone
from setup_common import LATEST_DOCKER_HUB_IMAGE

import argparse
from dataclasses import dataclass, fields
import subprocess
import time


@dataclass
class Params:
    add_gpu_to_staging_instance: bool = False
    docker_image: str = 'dshin83/alphazeroarcade:3.1.0'  # TODO: replace w/ LATEST_DOCKER_HUB_IMAGE
    image_name: str = ''
    image_family: str = Defaults.image_family
    machine_type: str = Defaults.machine_type

    # It's ok to over-provision the boot disk size, as it's only temporary, and as the long-term
    # storage cost of the image is only based on the archive size (i.e., the amount of space
    # occupied by the image).
    #
    # Bigger as better, as disk throughput scales with disk size. Empirically, we only actually use
    # about 17G.
    #
    # The cost of 200G of pd-ssd disk-space is about $34/month, with the timed used rounded up to
    # the nearest multiple of 0.1 seconds. Currently, this script takes about 17minutes, so this
    # translates to less than 2 cents in disk cost.
    boot_disk_size_gb: int = 200
    boot_disk_type: str = "pd-ssd"

    staging_instance_name: str = 'staging-instance'
    staging_zone: str = get_gcloud_zone()

    @staticmethod
    def create(args) -> 'Params':
        kwargs = {f.name: getattr(args, f.name) for f in fields(Params)}
        params = Params(**kwargs)
        if not params.image_name:
            params.image_name = time.strftime('i-%Y%m%d-%H%M%S')
        return params

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('create_image.py options')

        defaults = Params()
        group.add_argument('-g', '--add-gpu-to-staging-instance', action='store_true',
                           help='Add a GPU to the staging instance')
        group.add_argument('-D', '--docker-image', default=defaults.docker_image,
                           help='Docker image (default: %(default)s)')
        group.add_argument('-n', '--image-name', default=defaults.image_name,
                           help='Name of the image to create (default: i-YYYYmmdd-HHMMSS)')
        group.add_argument('-f', '--image-family', default=defaults.image_family,
                           help='Family of the image to create (default: %(default)s)')
        group.add_argument('-m', '--machine-type', default=defaults.machine_type,
                           help='Machine type to create the instance with (default: %(default)s)')
        group.add_argument('-s', '--boot-disk-size-gb', default=defaults.boot_disk_size_gb,
                           type=int, help='Boot disk size in GB (default: %(default)s)')
        group.add_argument('-t', '--boot-disk-type', default=defaults.boot_disk_type,
                           help=f'Boot disk type (default: %(default)s). See {Help.boot_disk_url}')
        group.add_argument('-i', '--staging-instance-name', default=defaults.staging_instance_name,
                           help='Name of the staging instance to create (default: %(default)s)')
        group.add_argument('-z', '--staging-zone', default=defaults.staging_zone,
                           help=f'Zone to create the staging instance in (default: %(default)s). '
                           f'See {Help.zone_url}')


def load_args() -> Params:
    parser = argparse.ArgumentParser()
    Params.add_args(parser)
    return Params.create(parser.parse_args())


def create_staging_instance(params: Params):
    """
    Step 1: Create a new VM instance with a Persistent Disk.
    """
    instance_name = params.staging_instance_name
    machine_type = params.machine_type
    zone = params.staging_zone
    boot_disk_size_gb = params.boot_disk_size_gb
    boot_disk_type = params.boot_disk_type

    print(f"Creating instance {instance_name}...")

    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        "--image-family=ubuntu-2204-lts",
        "--image-project=ubuntu-os-cloud",
        f"--boot-disk-size={boot_disk_size_gb}",
        f"--boot-disk-type={boot_disk_type}",
        "--boot-disk-auto-delete",
        "--maintenance-policy=TERMINATE",
    ]
    if params.add_gpu_to_staging_instance:
        cmd.extend([
            "--accelerator=type=nvidia-tesla-t4,count=1"
        ])

    subprocess.run(cmd, check=True)

    print(f"Instance {instance_name} created successfully!")
    print('Waiting until ssh access is available...')

    # do a simple loop with 5 second sleep to wait for ssh access
    timeout = 60
    sleep_time = 5
    success = False
    for _ in range(timeout // sleep_time):
        time.sleep(sleep_time)
        try:
            subprocess.run([
                "gcloud", "compute", "ssh", f"stager@{instance_name}", "--zone", zone, "--command",
                "exit",
            ], check=True, capture_output=True)
            success = True
            break
        except subprocess.CalledProcessError:
            pass

    if not success:
        print(f'❌ SSH access not available after {timeout} seconds! Exiting...')
        exit(1)
    else:
        print('✅ SSH access available!')


def configure_staging_instance(params: Params):
    """
    Step 2: Configure the staging instance

    NOTE: for ssh agent forwarding, we will want to run via:

    docker run -it --rm \
        -v $SSH_AUTH_SOCK:/ssh-agent \
        -e SSH_AUTH_SOCK=/ssh-agent \
        yourimage:latest
    """

    print(f"Configuring instance {params.staging_instance_name}...")

    # Copy the setup script to the staging instance
    subprocess.run([
        "gcloud", "compute", "scp", "gcp_staging_instance_setup.sh",
        f"stager@{params.staging_instance_name}:~/"
    ], check=True)

    # Run the setup script on the staging instance
    subprocess.run([
        "gcloud", "compute", "ssh", f"stager@{params.staging_instance_name}",
        "--command", "bash gcp_staging_instance_setup.sh",
    ], check=True)

    print('✅ Setup script ran successfully!')
    print(f"Pulling docker image {params.docker_image} on the staging instance...")

    # Pull the docker image on the staging instance
    subprocess.run([
        "gcloud", "compute", "ssh", f"stager@{params.staging_instance_name}",
        "--command", f"docker pull {params.docker_image}",
    ], check=True)

    # TODO: if params.add_gpu_to_staging_instance is True, perform some tests to verify that the GPU
    # is working correctly.

    # Copy the cleanup script to the staging instance
    subprocess.run([
        "gcloud", "compute", "scp", "gcp_staging_instance_cleanup.sh",
        f"stager@{params.staging_instance_name}:~/"
    ], check=True)

    subprocess.run([
        "gcloud", "compute", "ssh", f"stager@{params.staging_instance_name}",
        "--command", f"bash gcp_staging_instance_cleanup.sh"
    ], check=True)

    print('✅ Cleanup script ran successfully!')


def stop_staging_instance(params: Params):
    """
    Step 3: Stop the staging instance.
    """
    instance_name = params.staging_instance_name
    zone = params.staging_zone
    print(f"Stopping instance {instance_name}. This could take up to 5 minutes...")

    subprocess.run([
        "gcloud", "compute", "instances", "stop", instance_name,
        f"--zone={zone}"
    ], check=True)

    print(f"Instance {instance_name} stopped successfully!")


def create_custom_image(params: Params):
    """
    Step 4: Create an image from the stopped staging instance.

    Cmd:

    gcloud compute images create <CUSTOM_IMAGE_NAME> \
    --source-disk <STAGING_INSTANCE_NAME> \
    --source-disk-zone <YOUR_ZONE> \
    --family <OPTIONAL_IMAGE_FAMILY> \
    --description "Ubuntu 22.04 with NVIDIA driver, container toolkit, Docker, etc."

    Subsequent cmd to launch instance using this image:

    gcloud compute instances create <NEW_VM_NAME> \
    --zone=<YOUR_ZONE> \
    --image=<CUSTOM_IMAGE_NAME> \
    --machine-type=n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE
    """
    image_name = params.image_name
    image_family = params.image_family
    zone = params.staging_zone
    instance_name = params.staging_instance_name
    docker_image = params.docker_image

    print(f"Creating custom image {image_name}. This could take up to 5 minutes..")

    subprocess.run([
        "gcloud", "compute", "images", "create", image_name,
        f"--source-disk={instance_name}",
        f"--source-disk-zone={zone}",
        f"--family={image_family}",
        "--description", f"AlphaZeroArcade with Docker image {docker_image}"
    ], check=True)

    print(f"Custom image {image_name} created successfully!")


def delete_instance(params: Params):
    """
    Step 4: Delete the staging instance after imaging.
    """
    zone = params.staging_zone
    instance_name = params.staging_instance_name
    print(f"Deleting instance {instance_name}...")

    subprocess.run([
        "gcloud", "compute", "instances", "delete", instance_name,
        f"--zone={zone}", "--quiet",
    ], check=True)

    print(f"Instance {instance_name} deleted successfully!")


def main():
    params = load_args()

    create_staging_instance(params)
    configure_staging_instance(params)
    stop_staging_instance(params)
    create_custom_image(params)
    delete_instance(params)
    print('✅ Successfully created custom image!')


if __name__ == "__main__":
    main()
