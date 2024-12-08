#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from pull_docker_image import docker_pull
from setup_common import get_env_json, update_env_json, DOCKER_HUB_IMAGE

import os
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


def setup_output_dir():
    """
    Request user to set the output directory.
    """
    print('AlphaZeroArcade runs write (a lot of) data to disk. Please specify a directory')
    print('where that data will be written. This directory will be mounted into the docker')
    print('container. If you have a fast SSD, it is recommended to use that for the data')
    print('directory.')
    print('')

    env = get_env_json()
    cwd = os.getcwd()
    default_output_dir = env.get('OUTPUT_DIR', os.path.join(cwd, 'output'))
    prompt = f'Please enter the location of your output directory [{default_output_dir}]: '
    output_dir = input(prompt).strip()
    if not output_dir:
        output_dir = default_output_dir

    expanded_output_dir = os.path.expanduser(output_dir)
    try:
        os.makedirs(expanded_output_dir, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create output directory: {expanded_output_dir}")
        print(f"Error: {e}")
        raise SetupException()

    update_env_json({'OUTPUT_DIR': expanded_output_dir})
    print(f"✅ Successfully registered output directory: {output_dir}")

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
            print("    sudo groupadd docker (if it doesn't already exist)")
            print("    sudo usermod -aG docker $USER")
            print("Then log out and log back in.")
            print("Or run:")
            print("    newgrp docker")
        else:
            print("❌ Docker command failed for an unknown reason.")
            print("Error details:")
            print(stderr)


def main():
    print('*' * 80)
    print('Running AlphaZeroArcade setup wizard...')
    print('*' * 80)

    os.chdir(os.path.dirname(__file__))

    try:
        setup_output_dir()
        print('*' * 80)
        docker_pull(DOCKER_HUB_IMAGE)
        print('*' * 80)
        validate_nvidia_installation(DOCKER_HUB_IMAGE)
        print('*' * 80)
        check_docker_permissions()
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
