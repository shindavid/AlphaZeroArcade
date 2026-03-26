#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from pull_docker_image import docker_pull
from setup_common import get_env_json, update_env_json, is_subpath, LATEST_DOCKER_HUB_IMAGE

import os
import shutil
import subprocess


def print_green(text):
    green_start = "\033[32m"
    reset = "\033[0m"
    print(f"{green_start}{text}{reset}")


def print_red(text):
    red_start = "\033[31m"
    reset = "\033[0m"
    print(f"{red_start}{text}{reset}")


def run(cmd: str, print_cmd=False, print_output=False):
    """
    Runs the given cmd. If successful, returns None. Else, returns the proc object, from
    which the stdout/stderr can be read.
    """
    if print_cmd:
        print(cmd)

    if print_output:
        stdout = None
        stderr = None
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    p = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
    p.wait()
    if p.returncode:
        return p
    return None


class SetupException(Exception):
    # When caught, will only print the first argument, not the full traceback
    pass


class VerboseSetupException(Exception):
    # When caught, will print the full traceback
    pass


def validate_nvidia_driver():
    """Validate that the NVIDIA driver is installed and working."""
    print('Validating NVIDIA driver installation...')
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print("✅ NVIDIA driver is installed and working.")
        return
    else:
        print("❌ NVIDIA driver validation failed.")
        print(result.stderr.decode())
        print('')
        print("Please check NVIDIA website for driver installation instructions.")
        raise SetupException()


def validate_nvidia_installation(image):
    """Validate that NVIDIA components are installed and functional."""
    print('Validating NVIDIA installation...')
    test_cmd = ["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi"]
    result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("✅ NVIDIA Container Toolkit is installed and GPU is accessible in Docker.")
    else:
        # first check NVIDIA driver
        validate_nvidia_driver()

        # if we got here, the driver is installed but the container toolkit is not working
        print("❌ NVIDIA Container Toolkit validation failed.")
        print(result.stderr)
        print("Please read here for Container Toolkit installation instructions:")
        print('')
        print('https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html')
        print('')
        print('Likely applicable sections:')
        print('  - Installing with Apt')
        print('  - Configuring Docker')
        raise SetupException()


def setup_mount_dir():
    """
    Request user to set the output directory.
    """
    print("AlphaZeroArcade writes a significant amount of data to disk and requires persistent storage.")
    print("Please specify a directory where this data should be stored. The directory will be mounted")
    print("into the Docker container. For best performance, use a fast SSD if available.")
    print("It's required to place this directory **outside** the project repo.")

    env = get_env_json()
    cwd = os.getcwd()
    default_mount_dir = env.get('MOUNT_DIR', os.path.expanduser('~/AlphaZeroArcade-mount'))

    while True:
        prompt = f'Please enter the location of your mount directory [{default_mount_dir}]: '
        mount_dir = input(prompt).strip()
        if not mount_dir:
            mount_dir = default_mount_dir

        if is_subpath(mount_dir, cwd):
            print_red(f"❌ The mount directory cannot be inside the project repo: {cwd}")
            continue
        else:
            break

    expanded_mount_dir = os.path.expanduser(mount_dir)
    try:
        os.makedirs(expanded_mount_dir, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create mount directory: {expanded_mount_dir}")
        print(f"Error: {e}")
        raise SetupException()

    update_env_json({'MOUNT_DIR': expanded_mount_dir})
    print(f"✅ Successfully registered mount directory: {expanded_mount_dir}")


def check_docker_permissions():
    """Check if the user can run Docker commands without sudo."""
    print('Checking if you have permission to run Docker commands without sudo...')

    result = subprocess.run(['docker', 'ps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print("✅ You have permission to run Docker commands without sudo.")
    else:
        # Check for permission-related errors
        stderr = result.stderr.decode()
        if "permission denied" in stderr.lower():
            print("❌ You do not have permission to run Docker commands without sudo.")
            print("To fix this, add your user to the Docker group by running:")
            print("    sudo usermod -aG docker $USER")
            print("Then, log out and log back in.")
        else:
            print("❌ Docker command failed for an unknown reason.")
            print("Error details:")
            print(stderr)
        raise SetupException()


SYZYGY_DOWNLOAD_BASE = 'https://tablebase.lichess.ovh/tables/standard/'
SYZYGY_DOWNLOAD_SUBDIRS = ['3-4-5-wdl/', '3-4-5-dtz/']

# Representative 5-piece tablebase files used to verify completeness.
SYZYGY_5PIECE_SENTINEL_FILES = [
    'KQRvKR.rtbw',
    'KBBvKN.rtbw',
    'KPPvKP.rtbw',
]

SYZYGY_COMMON_LOCATIONS = [
    os.path.expanduser('~/syzygy'),
    '/usr/share/syzygy',
    '/usr/local/share/syzygy',
]


def has_5piece_syzygy(path):
    """Check if a directory contains 5-piece Syzygy tablebases."""
    if not os.path.isdir(path):
        return False
    return all(os.path.isfile(os.path.join(path, f)) for f in SYZYGY_5PIECE_SENTINEL_FILES)


def looks_like_syzygy_dir(path):
    """Check if a directory appears to contain any Syzygy tablebases."""
    if not os.path.isdir(path):
        return False
    return any(f.endswith('.rtbw') for f in os.listdir(path))


def find_existing_syzygy():
    """Search common locations for existing Syzygy tablebases."""
    for loc in SYZYGY_COMMON_LOCATIONS:
        if looks_like_syzygy_dir(loc):
            return loc
    return None


def download_syzygy(target_dir):
    """Download 3-4-5 piece Syzygy tablebases via wget."""
    if not shutil.which('wget'):
        print_red('❌ wget is required to download Syzygy tables but was not found on PATH.')
        raise SetupException()

    os.makedirs(target_dir, exist_ok=True)
    print(f'Downloading 3-4-5 piece Syzygy tablebases to: {target_dir}')
    print('This is approximately 1 GB and may take several minutes...')
    print()

    for subdir in SYZYGY_DOWNLOAD_SUBDIRS:
        url = SYZYGY_DOWNLOAD_BASE + subdir
        cmd = (
            f'wget --recursive --no-parent --no-host-directories --cut-dirs=3 '
            f'-e robots=off --accept "*.rtbw,*.rtbz" --no-verbose '
            f'-P "{target_dir}" '
            f'{url}'
        )
        result = run(cmd, print_output=True)
        if result is not None:
            print_red(f'❌ Syzygy download failed for {subdir}')
            raise SetupException()

    if not has_5piece_syzygy(target_dir):
        print_red('❌ Download completed but 5-piece tables not found. The download may be incomplete.')
        raise SetupException()


def setup_syzygy():
    """Setup Syzygy endgame tablebases for chess."""
    print('Checking for Syzygy endgame tablebases...')

    env = get_env_json()
    syzygy_path = env.get('SYZYGY_PATH')

    # If already configured and valid, skip
    if syzygy_path and has_5piece_syzygy(syzygy_path):
        print(f'✅ Syzygy 5-piece tables already set up at: {syzygy_path}')
        return

    print('Syzygy tablebases not found.')
    print('')
    print('Syzygy endgame tablebases provide perfect play information for chess endgames.')
    print('The 3-4-5 piece tables (~1 GB) are required by this project if you want to use it for chess.')
    print()

    # Offer to skip (relevant if user isn't planning to run chess)
    response = input('Skip Syzygy setup? [y/N]: ').strip().lower()
    if response in ('y', 'yes'):
        print('Skipping Syzygy setup. You can re-run the setup wizard later to set it up.')
        return

    # Auto-detect in common locations
    target_path = None
    if not syzygy_path:
        found = find_existing_syzygy()
        if found:
            print(f'Found existing Syzygy tables at: {found}')
            response = input('Use this location? [Y/n]: ').strip().lower()
            if response in ('', 'y', 'yes'):
                target_path = found

    # If not auto-detected (or user declined), prompt for existing location
    while target_path is None:
        user_path = input(
            'If you already have Syzygy downloaded, specify the path [press Enter if not]: '
        ).strip()

        if user_path:
            target_path = os.path.expanduser(user_path)
            if not os.path.isdir(target_path):
                print_red(f'❌ The specified path does not exist: {target_path}')
                target_path = None
        else:
            # User doesn't have Syzygy — ask where to download
            default_path = syzygy_path or os.path.expanduser('~/syzygy')
            target_path = input(
                f'Syzygy tablebases will be downloaded (~1 GB). '
                f'Please specify installation location [{default_path}]: '
            ).strip()
            if not target_path:
                target_path = default_path
            target_path = os.path.expanduser(target_path)

    # Check 5-piece completeness; download if needed
    if has_5piece_syzygy(target_path):
        update_env_json({'SYZYGY_PATH': target_path})
        print(f'✅ Syzygy 5-piece tables found at: {target_path}')
        return

    download_syzygy(target_path)
    update_env_json({'SYZYGY_PATH': target_path})
    print(f'✅ Syzygy path set to: {target_path}')


def main():
    print('*' * 80)
    print('Running AlphaZeroArcade setup wizard...')
    print('*' * 80)

    os.chdir(os.path.dirname(__file__))

    try:
        setup_mount_dir()
        print('*' * 80)
        setup_syzygy()
        print('*' * 80)
        check_docker_permissions()
        print('*' * 80)
        docker_pull(LATEST_DOCKER_HUB_IMAGE)
        print('*' * 80)
        validate_nvidia_installation(LATEST_DOCKER_HUB_IMAGE)
    except KeyboardInterrupt:
        print('')
        print('Setup wizard was interrupted. Please try again.')
        return
    except SetupException as e:
        for arg in e.args:
            print('*' * 80)
            print(arg)
        # Call site should print further details
        return
    except VerboseSetupException as e:
        print('*' * 80)
        raise
    except:
        print('*' * 80)
        print('Setup wizard failed unexpectedly! See below for details.')
        print('*' * 80)
        raise


if __name__ == '__main__':
    main()
