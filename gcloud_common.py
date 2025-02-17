import subprocess


class Help:
    zone_url = 'https://cloud.google.com/compute/docs/regions-zones#available'
    preemptible_url = 'https://cloud.google.com/compute/docs/instances/preemptible'
    machine_type_url = 'https://cloud.google.com/compute/docs/machine-resource'
    gpu_type_url = 'https://cloud.google.com/compute/docs/gpus'
    boot_disk_url = 'https://cloud.google.com/compute/docs/disks/create-root-persistent-disks'


def get_gcloud_project():
    return subprocess.check_output(["gcloud", "config", "get-value", "project"], text=True).strip()
