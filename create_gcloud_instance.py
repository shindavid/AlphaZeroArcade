#!/usr/bin/env python3
from gcloud.common import get_gcloud_project

from google.cloud import compute_v1

import argparse
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class Params:
    name: Optional[str] = None
    preemptible: bool = False
    zone: str = 'asia-northeast3-b'  # Seoul

    machine_type: str = "n1-standard-8"
    gpu_type: str = "nvidia-tesla-t4"
    boot_disk_size_gb: int = 100
    local_ssd_count: int = 1

    @staticmethod
    def create(args) -> 'Params':
        kwargs = {f.name: getattr(args, f.name) for f in fields(Params)}
        return Params(**kwargs)

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('create_instance.py options')

        zone_url = 'https://cloud.google.com/compute/docs/regions-zones#available'
        preemptible_url = 'https://cloud.google.com/compute/docs/instances/preemptible'
        machine_type_url = 'https://cloud.google.com/compute/docs/machine-resource'
        gpu_type_url = 'https://cloud.google.com/compute/docs/gpus'
        boot_disk_url = 'https://cloud.google.com/compute/docs/disks/create-root-persistent-disks'

        defaults = Params()
        group.add_argument('-n', '--name', help='Name of the instance to create (required)')
        group.add_argument(
            '--preemptible', action='store_true',
            help=f'If set, create a preemptible instance. See {preemptible_url}')

        group.add_argument(
            '-z', '--zone', default=defaults.zone,
            help=f'Zone to create the instance in (default: %(default)s). See {zone_url}')

        group.add_argument(
            '-m', '--machine-type', default=defaults.machine_type,
            help=f'Machine type to create the instance with (default: %(default)s). See {machine_type_url}')

        group.add_argument(
            '-g', '--gpu-type', default=defaults.gpu_type,
            help=f'GPU type to create the instance with (default: %(default)s). See {gpu_type_url}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", '--zone', default=LOCAL_DOCKER_IMAGE,
                        help='Local docker image name (default: %(default)s)')
    parser.add_argument("-i", '--docker-hub-image', default=DOCKER_HUB_IMAGE,
                        help='Docker Hub image name, without tag (default: %(default)s)')
    parser.add_argument("-t", '--tag',
                        help='Comma-separated tags to push to Docker Hub '
                        '(default: "{VERSION},latest", where VERSION is the version label '
                        'of the local image)')
    return parser.parse_args()


STARTUP_SCRIPT = """#!/bin/bash

# Create devuser
useradd -m -s /bin/bash devuser
echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
mkdir -p /home/devuser/.ssh
chmod 700 /home/devuser/.ssh
cp /home/$USER/.ssh/authorized_keys /home/devuser/.ssh
chown -R devuser:devuser /home/devuser/.ssh

# Detect and mount Persistent Disk dynamically
PDISK=$(readlink -f /dev/disk/by-id/google-my-backup-disk)
if [ -b "$PDISK" ]; then
    sudo mkdir -p /workspace
    sudo mount "$PDISK" /workspace
    sudo chmod -R 777 /workspace
    sudo chown -R devuser:devuser /workspace
    echo "$PDISK /workspace ext4 defaults 0 0" | sudo tee -a /etc/fstab
fi

# Detect and mount Local SSD dynamically
LSSD=$(lsblk -o NAME,MODEL | grep "NVMe" | awk '{print "/dev/"$1}')
if [ -b "$LSSD" ]; then
    sudo mkfs.ext4 -F "$LSSD"
    sudo mkdir -p /scratch
    sudo mount "$LSSD" /scratch
    sudo chmod -R 777 /scratch
    sudo chown -R devuser:devuser /scratch
    echo "$LSSD /scratch ext4 defaults 0 0" | sudo tee -a /etc/fstab
fi
"""


def create_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str = "n1-standard-8",
    gpu_type: str = "nvidia-tesla-t4",
    image_family: str = "ubuntu-2204-lts",
    image_project: str = "ubuntu-os-cloud",
    boot_disk_size_gb: int = 100,
    local_ssd_count: int = 1,
    preemptible: bool = True,
    startup_script: str = """#!/bin/bash
        # Create devuser
        useradd -m -s /bin/bash devuser
        echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
        mkdir -p /home/devuser/.ssh
        chmod 700 /home/devuser/.ssh
        cp /home/$USER/.ssh/authorized_keys /home/devuser/.ssh
        chown -R devuser:devuser /home/devuser/.ssh

        # Ensure persistent disk (/workspace/) is mounted
        if ! mountpoint -q /workspace/; then
          sudo mkdir -p /workspace
          sudo mount /dev/sdb /workspace
          sudo chmod -R 777 /workspace
          echo "/dev/sdb /workspace ext4 defaults 0 0" | sudo tee -a /etc/fstab
        fi

        # Ensure Local SSD (/scratch/) is mounted
        if ! mountpoint -q /scratch/; then
          sudo mkfs.ext4 -F /dev/nvme0n1
          sudo mkdir -p /scratch
          sudo mount /dev/nvme0n1 /scratch
          echo "/dev/nvme0n1 /scratch ext4 defaults 0 0" | sudo tee -a /etc/fstab
        fi

        # Ensure devuser owns /workspace/
        sudo chown -R devuser:devuser /workspace/
    """
):
    instance_client = compute_v1.InstancesClient()

    disk = compute_v1.AttachedDisk()
    disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=f"projects/{image_project}/global/images/family/{image_family}",
        disk_size_gb=boot_disk_size_gb,
        disk_type=f"zones/{zone}/diskTypes/pd-ssd"
    )
    disk.auto_delete = True
    disk.boot = True

    network_interface = compute_v1.NetworkInterface()
    network_interface.name = "default"

    accelerator = compute_v1.AcceleratorConfig(
        accelerator_type=f"zones/{zone}/acceleratorTypes/{gpu_type}",
        accelerator_count=1
    )

    local_ssd_disks = [
        compute_v1.AttachedDisk(
            type_="SCRATCH",
            interface="NVME",
            auto_delete=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                disk_type=f"zones/{zone}/diskTypes/local-ssd"
            )
        )
        for _ in range(local_ssd_count)
    ]

    instance = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk] + local_ssd_disks,
        network_interfaces=[network_interface],
        guest_accelerators=[accelerator],
        scheduling=compute_v1.Scheduling(
            on_host_maintenance="TERMINATE",
            preemptible=preemptible
        ),
        metadata=compute_v1.Metadata(items=[
            compute_v1.Items(key="startup-script", value=startup_script)
        ])
    )

    operation = instance_client.insert(
        project=project_id,
        zone=zone,
        instance_resource=instance
    )

    print(f"Creating instance {instance_name}...")
    operation.result()
    print(f"Instance {instance_name} created successfully!")

# Example usage:
create_instance(
    project_id="your-project-id",
    zone="asia-northeast3-b",
    instance_name="a0a-compute-instance"
)
