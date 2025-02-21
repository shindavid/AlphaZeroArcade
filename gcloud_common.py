import subprocess
from typing import Optional


class Defaults:
    username = 'devuser'
    image_family = 'alphazero-arcade'
    a0a_project = 'nimble-chess-450406-q7'
    machine_type = 'a2-highgpu-1g'
    gpu_type = None


class Help:
    zone_url = 'https://cloud.google.com/compute/docs/regions-zones#available'
    preemptible_url = 'https://cloud.google.com/compute/docs/instances/preemptible'
    machine_type_url = 'https://cloud.google.com/compute/docs/machine-resource'
    gpu_type_url = 'https://cloud.google.com/compute/docs/gpus'
    boot_disk_url = 'https://cloud.google.com/compute/docs/disks/create-root-persistent-disks'
    gcloud_install_url = 'https://cloud.google.com/sdk/docs/install'
    gcping_url = 'https://www.gcping.com/'
    quotas_url = 'https://cloud.google.com/compute/quotas'
    quotas_iam_admin_url = 'https://console.cloud.google.com/iam-admin/quotas'


def get_config_value(key: str) -> Optional[str]:
    try:
        cmd = ["gcloud", "config", "get-value", key]
        return subprocess.check_output(cmd, text=True).strip()
    except:
        return None


def get_gcloud_project() -> Optional[str]:
    return get_config_value("project")


def get_gcloud_zone() -> Optional[str]:
    return get_config_value("compute/zone")
