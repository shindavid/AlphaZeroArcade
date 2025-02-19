import subprocess


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


def get_gcloud_project():
    return subprocess.check_output(["gcloud", "config", "get-value", "project"], text=True).strip()
