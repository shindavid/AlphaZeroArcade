#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from gcloud_common import Help

import json
import os
import subprocess


class SetupException(Exception):
    pass


def check_gcloud_installation():
    """
    Check that gcloud is installed.
    """
    result = subprocess.run(["gcloud", "version"], capture_output=True)
    if result.returncode == 0:
        print('✅ gcloud is installed.')
        return

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

    default_project_name = 'AlphaZeroArcade'

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
    if result1.returncode != 0:
        while True:
            region = input('Enter region: ')
            if not region:
                continue
            break
    else:
        default_region = result1.stdout.decode().strip()
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
    if result2.returncode != 0:
        print('Please set a default zone.')

        suggested_zone = region + '-a'
        zone = input(f'Enter zone (default: {suggested_zone}): ')
        if not zone:
            zone = suggested_zone
    else:
        default_zone = result2.stdout.decode().strip()
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
