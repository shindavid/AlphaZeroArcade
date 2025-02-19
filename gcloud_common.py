import subprocess
from typing import Optional


class Defaults:
    username = 'devuser'
    image_family = 'alphazero-arcade'
    machine_type = 'n1-standard-8'
    gpu_type = 'nvidia-tesla-t4'


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
