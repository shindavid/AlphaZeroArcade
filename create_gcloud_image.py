#!/usr/bin/env python3

"""
Creates a custom GCP image, to be used for instance creation. Only the GCP project owner (dshin)
needs to run this script. Other users, as long as they are part of the AlphaZeroArcade GCP project,
can launch instances from the custom image.
"""

from gcloud_common import Help
from setup_common import LATEST_DOCKER_HUB_IMAGE

from google.cloud import compute_v1
from googleapiclient import discovery
import google.auth
from google.oauth2 import service_account

import argparse
from dataclasses import dataclass, fields
import time


@dataclass
class Params:
    add_gpu_to_staging_instance: bool = False
    docker_image: str = LATEST_DOCKER_HUB_IMAGE
    image_name: str = ''
    image_family: str = 'alphazero-arcade'

    machine_type: str = "n1-standard-8"
    boot_disk_size_gb: int = 100
    boot_disk_type: str = "pd-ssd"

    staging_instance_name: str = 'staging-instance'
    staging_zone = 'asia-northeast3-b'  # Seoul

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

    # instance_client = compute_v1.InstancesClient()

    # disk = compute_v1.AttachedDisk()
    # disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
    #     source_image="projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts",
    #     disk_size_gb=boot_disk_size_gb,
    #     disk_type=f"zones/{zone}/diskTypes/{boot_disk_type}",
    # )
    # disk.auto_delete = True
    # disk.boot = True

    # network_interface = compute_v1.NetworkInterface()
    # network_interface.name = "default"

    # instance = compute_v1.Instance(
    #     name=instance_name,
    #     machine_type=f"zones/{zone}/machineTypes/{machine_type}",
    #     disks=[disk],
    #     network_interfaces=[network_interface],
    # )

    # operation = instance_client.insert(
    #     project=project, zone=zone, instance_resource=instance
    # )
    # operation.result()
    # print(f"Instance {instance_name} created successfully!")

    # credentials = service_account.Credentials.from_service_account_file(
    #     'path/to/your-service-account-key.json'
    # )

    credentials, project = google.auth.default()
    compute = discovery.build('compute', 'v1', credentials=credentials)

    # --machine-type=n1-standard-4
    machine_type = f"zones/{zone}/machineTypes/n1-standard-4"

    # OPTIONAL: Add a GPU to the instance - useful for ad-hoc testing of staging instance
    # --accelerator type=nvidia-tesla-t4,count=1
    guest_accelerators = []
    if params.add_gpu_to_staging_instance:
        guest_accelerators = [{
            "acceleratorType": f"projects/{project}/zones/{zone}/acceleratorTypes/nvidia-tesla-t4",
            "acceleratorCount": 1
        }]

    # --maintenance-policy=TERMINATE
    scheduling = {
        "onHostMaintenance": "TERMINATE",
        "automaticRestart": False
    }

    # --image-family=ubuntu-2204-lts
    # --image-project=ubuntu-os-cloud
    # --boot-disk-auto-delete
    # --boot-disk-size=<boot_disk_size_gb>
    # --boot-disk-type=pd-ssd
    disks = [
        {
            "boot": True,
            "autoDelete": True,
            "initializeParams": {
                "sourceImage": "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts",
                "diskSizeGb": boot_disk_size_gb,
                "diskType": f"projects/{project}/zones/{zone}/diskTypes/{boot_disk_type}"
            }
        }
    ]

    # Minimal network interface to give the instance internet access
    network_interfaces = [
        {
            "network": "global/networks/default",
            "accessConfigs": [
                {"type": "ONE_TO_ONE_NAT", "name": "External NAT"}
            ]
        }
    ]

    # Build the instance config
    instance_body = {
        "name": instance_name,
        "machineType": machine_type,
        "guestAccelerators": guest_accelerators,
        "scheduling": scheduling,
        "disks": disks,
        "networkInterfaces": network_interfaces
        # Optionally add 'metadata', 'serviceAccounts', etc. here
    }

    request = compute.instances().insert(
        project=project,
        zone=zone,
        body=instance_body
    )
    response = request.execute()

    print("Instance creation response:", response)


def configure_staging_instance(params: Params):
    """
    Step 2: Configure the staging instance

    - Set up swap space
    - Docker setup
    """
    print('TODO: Configure the staging instance as needed.')


def stop_staging_instance(params: Params):
    """
    Step 3: Stop the staging instance.
    """
    project = params.project
    instance_name = params.staging_instance_name
    zone = params.staging_zone
    print(f"Stopping instance {instance_name}. This could take up to 5 minutes...")

    instance_client = compute_v1.InstancesClient()
    operation = instance_client.stop(project=project, zone=zone, instance=instance_name)
    operation.result()

    print(f"Instance {instance_name} stopped successfully!")


def create_custom_image(params: Params):
    """
    Step 4: Create an image from the stopped staging instance.
    """
    project = params.project
    image_name = params.image_name
    image_family = params.image_family
    zone = params.staging_zone
    instance_name = params.staging_instance_name

    print(f"Creating custom image {image_name}. This could take up to 5 minutes..")

    image_client = compute_v1.ImagesClient()
    image = compute_v1.Image(
        name=image_name,
        source_disk=f"projects/{project}/zones/{zone}/disks/{instance_name}",
        family=image_family,
    )

    operation = image_client.insert(project=project, image_resource=image)
    operation.result()

    print(f"Custom image {image_name} created successfully!")


def delete_instance(params: Params):
    """
    Step 4: Delete the staging instance after imaging.
    """
    project = params.project
    zone = params.staging_zone
    instance_name = params.staging_instance_name
    print(f"Deleting instance {instance_name}...")

    instance_client = compute_v1.InstancesClient()
    operation = instance_client.delete(project=project, zone=zone, instance=instance_name)
    operation.result()

    print(f"Instance {instance_name} deleted successfully!")


def delete_disk(params: Params):
    """
    Step 5: Delete the Persistent Disk after imaging.
    """
    project = params.project
    zone = params.staging_zone
    instance_name = params.staging_instance_name
    print(f"Deleting persistent disk {instance_name}...")

    disk_client = compute_v1.DisksClient()
    operation = disk_client.delete(project=project, zone=zone, disk=instance_name)
    operation.result()

    print(f"Persistent disk {instance_name} deleted successfully!")


def main():
    params = load_args()

    create_staging_instance(params)
    configure_staging_instance(params)
    # stop_staging_instance(params)
    # create_custom_image(params)
    # delete_instance(params)
    # print('✅ Successfully created custom image!')

if __name__ == "__main__":
    main()
