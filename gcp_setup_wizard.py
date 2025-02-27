#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from gcloud_common import Help
from setup_common import get_env_json, update_env_json

import json
import os
import subprocess


class SetupException(Exception):
    pass


def check_gcloud_installation():
    """
    Check that gcloud is installed.
    """
    try:
        result = subprocess.run(["gcloud", "version"], capture_output=True)
        if result.returncode == 0:
            print('✅ gcloud is installed.')
            return
    except FileNotFoundError:
        pass

    print("❌ gcloud is not installed.")
    print("Please install gcloud before proceeding.")
    print(f"See: {Help.gcloud_install_url}")
    raise SetupException()


def gcloud_auth_login():
    """
    Check that gcloud is authenticated. If not, authenticate.
    """
    result = subprocess.run(["gcloud", "auth", "list"], capture_output=True)
    if result.returncode == 0:
        print('✅ gcloud is authenticated.')
        return

    print('Performing one-time gcloud authentication...')
    result = subprocess.run(["gcloud", "auth", "login"], capture_output=True)
    if result.returncode == 0:
        print('✅ gcloud authenticated successfully.')
        return

    print('❌ gcloud authentication failed.')
    print(result.stderr.decode())
    print('')
    print('Please authenticate with "gcloud auth login" and then rerun this wizard.')
    raise SetupException()


def gcloud_check_project():
    """
    Set default gcloud project.
    """

    default_project_name = 'alphazeroarcade'

    result = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True)
    if result.returncode == 0:
        default_project_name = result.stdout.decode().strip()

    project = input(f'Enter project name (default: {default_project_name}): ')
    if not project:
        project = default_project_name
    else:
        # TODO: validate project name
        pass

    result = subprocess.run(["gcloud", "projects", "describe", project], capture_output=True)
    if result.returncode != 0:
        print(f'Project {project} does not exist.')
        create_project = input(f'Would you like to create project {project}? (Y/n): ')
        if create_project.strip().lower() not in ('y', ''):
            print('Ok, good-bye!')
            raise SetupException()
        cmd = ["gcloud", "projects", "create", project, "--set-as-default"]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f'❌ Failed to create project {project}.')
            print(result.stderr.decode())
            raise SetupException()

        print(f'✅ Project {project} created successfully.')
    else:
        # project exists, but is not set as default
        cmd = ["gcloud", "config", "set", "project", project]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f'❌ Failed to set project {project} as default.')
            print(result.stderr.decode())
            raise SetupException()
        else:
            print(f'✅ Project {project} set as default.')


def gcloud_check_region_and_zone():
    """
    Set default gcloud region/zone.
    """
    print('Please set a default region near you.')
    print('This link may be helpful in making this choice:')
    print('')
    print(f'  {Help.gcping_url}')
    print('')

    result1 = subprocess.run(["gcloud", "config", "get-value", "compute/region"], capture_output=True)
    default_region = ''
    if result1.returncode == 0:
        default_region = result1.stdout.decode().strip()

    if not default_region:
        while True:
            region = input('Enter region: ')
            if not region:
                continue
            if len(region) > 1 and region[-2] == '-' and region[-1].isalpha():
                print('It appears you entered a zone instead of a region.')
                print('Please enter a region (e.g., us-central1).')
                continue
            break
    else:
        region = input(f'Enter region (default: {default_region}): ')
        if not region:
            region = default_region

    cmd = ["gcloud", "config", "set", "compute/region", region]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f'❌ Failed to set region {region}.')
        print(result.stderr.decode())
        raise SetupException()
    else:
        print(f'✅ Region {region} set as default.')

    result2 = subprocess.run(["gcloud", "config", "get-value", "compute/zone"], capture_output=True)
    default_zone = ''
    if result2.returncode == 0:
        default_zone = result2.stdout.decode().strip()

    if not default_zone:
        print('Please set a default zone.')

        suggested_zone = region + '-a'
        zone = input(f'Enter zone (default: {suggested_zone}): ')
        if not zone:
            zone = suggested_zone
    else:
        if not default_zone.startswith(region):
            default_zone = region + '-a'

        zone = input(f'Enter zone (default: {default_zone}): ')
        if not zone:
            zone = default_zone

    cmd = ["gcloud", "config", "set", "compute/zone", zone]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f'❌ Failed to set zone {zone}.')
        print(result.stderr.decode())
        raise SetupException()
    else:
        print(f'✅ Zone {zone} set as default.')


def gcloud_check_gpu_quota():
    """
    Checks whether the user has a nonzero GPU quota. If not, walks the user through the process of
    requesting a GPU quota increase.
    """

    cmd = ["gcloud", "compute", "project-info", "describe", '--format="json"']
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print('❌ Unexpected error while checking gpu quota.')
        print(result.stderr.decode())
        raise SetupException()

    project_info_json = result.stdout.decode().strip()
    project_info = json.loads(project_info_json)
    quotas = project_info.get('quotas', [])
    gpu_all_regions_quotas = [q for q in quotas if q['metric'] == 'GPUS_ALL_REGIONS']
    limit = 0
    if gpu_all_regions_quotas:
        assert len(gpu_all_regions_quotas) == 1
        gpu_all_regions_quota = gpu_all_regions_quotas[0]
        limit = float(gpu_all_regions_quota['limit'])

    if limit > 0:
        print(f'✅ You have a global GPU quota of {limit} GPUs.')
        return

    print('You do not have a global GPU quota set.')
    print('Please request a GPU quota increase using the following link:')
    print('')
    print(Help.quotas_url)
    print('')
    print('This is unfortunately a manual process, and it is hard to describe how to navigate ')
    print('the web interface precisely. Here is my best attempt:')
    print('')
    print(f'1. Go to the GCP Quotas dashboard: {Help.quotas_iam_admin_url}')
    print('')
    print('2. Click the "Filter" textbox, in the open space just to the right of "Filter".')
    print('')
    print('3. Select "Service" in the dropdown, and then "Compute Engine API" in the next dropdown.')
    print('')
    print('4. Select "Name" in the next dropdown, and then "GPUs (all regions)" in the next dropdown.')
    print('   The dropdown won\'t show "GPUs (all regions)" until you type enough of it out.')
    print('')
    print('5. You should see a single search result below the textbox. Click the triple-dots menu')
    print('   on the right side of the search result, and then click "Edit Quota".')
    print('')
    print('6. A pop-up will appear on the right-side. Enter "1" for "New value", enter a short')
    print('   reason for the quota increase, and then click "Next".')
    print('')
    print('Officially, the quota increase can take up to 48 hours to process, but it appears')
    print('to be instantaneous in practice (at least when setting the value only to 1). Check your')
    print('email for a notification that the quota increase has been approved.')
    print('')
    print('After you have received the email, you can rerun this setup wizard to continue.')
    raise SetupException()


def gcloud_check_persistent_disk():
    """
    Configures the persistent disk the user will use for this project.
    """
    env = get_env_json()
    default_disk_name = env.get('GCP_PERSISTENT_DISK', 'alphazero-arcade-disk')
    prompt = f'Please enter the name of your gcloud disk [{default_disk_name}]: '
    disk_name = input(prompt).strip()
    if not disk_name:
        disk_name = default_disk_name

    cmd = ["gcloud", "compute", "disks", "describe", disk_name]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f'Disk {disk_name} does not exist.')
        print('')
        print('You are now required to create a gcloud disk for long-term data storage.')
        print('It will cost approximately $0.04 per GB per month.')
        print('Time usage is rounded up to the nearest 0.1 seconds.')
        print('In the future, we may make this disk optional, but for now, it is required.')
        print('')
        create_disk = input(f'Would you like to create disk {disk_name}? (Y/n): ')
        if create_disk.strip().lower() not in ('y', ''):
            print('Ok, good-bye!')
            raise SetupException()

        default_disk_size_gb = 2000
        disk_size_gb = input(f'Enter disk size in GB (default: {default_disk_size_gb}): ')
        if not disk_size_gb:
            disk_size_gb = default_disk_size_gb

        cmd = ["gcloud", "compute", "disks", "create", disk_name, f'--size={disk_size_gb}GB']
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f'❌ Failed to create disk {disk_name}.')
            print(result.stderr.decode())
            raise SetupException()

        print(f'✅ Disk {disk_name} created successfully.')

    # TODO: validate that the disk zone matches the default zone

    update_env_json({'GCP_PERSISTENT_DISK': disk_name})
    print(f'✅ Disk {disk_name} set as default.')


def main():
    print('*' * 80)
    print('Running AlphaZeroArcade GCP setup wizard...')
    print('*' * 80)

    os.chdir(os.path.dirname(__file__))

    try:
        check_gcloud_installation()
        gcloud_auth_login()
        gcloud_check_project()
        gcloud_check_region_and_zone()
        gcloud_check_gpu_quota()
        gcloud_check_persistent_disk()
        print('*' * 80)
        print('✅ GCP setup wizard completed successfully!')
    except KeyboardInterrupt:
        print('')
        print('❌ GCP setup wizard was interrupted. Please try again.')
        return
    except SetupException as e:
        return
    except:
        print('*' * 80)
        print('❌ GCP setup wizard failed unexpectedly! See below for details.')
        print('*' * 80)
        raise


if __name__ == '__main__':
    main()
