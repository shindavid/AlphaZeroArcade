#!/usr/bin/env python3
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


def build_docker_image(env_sh_lines):
    """
    Request user to select a docker image name, using "a0a" as the default.
    If the image does not exist, create it.
    """
    print('AlphaZeroArcade runs in a docker container.')
    print('')

    # Check that docker is installed:
    proc = subprocess.run(['docker', '--version'], capture_output=True)
    if proc.returncode:
        raise SetupException('docker is not installed. Please install it first.')

    default_image_name = os.environ.get('A0A_DOCKER_IMAGE', 'a0a')
    image_name = input(f'Please enter a docker image name [{default_image_name}]: ')
    if not image_name:
        image_name = default_image_name

    # TODO: validate image_name
    env_sh_lines.append(f'export A0A_DOCKER_IMAGE={image_name}')

    print(f'Building docker image {image_name}...')

    cmd = f'docker build -t {image_name} docker-setup/'
    if run(cmd, print_cmd=True, print_output=True):
        print(f"❌ Failed to build docker image {image_name}.")
        raise VerboseSetupException()

    print(f"✅ Successfully built docker image {image_name}!")
    return image_name


def setup_output_dir(env_sh_lines):
    """
    Request user to set the output directory.
    """
    print('AlphaZeroArcade runs write (a lot of) data to disk. Please specify a directory')
    print('where that data will be written. This directory will be mounted into the docker')
    print('container. If you have a fast SSD, it is recommended to use that for the data')
    print('directory.')
    print('')

    cwd = os.getcwd()
    default_output_dir = os.environ.get('A0A_OUTPUT_DIR', os.path.join(cwd, 'output'))
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
    env_sh_lines.append(f'export A0A_OUTPUT_DIR={expanded_output_dir}')
    print(f"✅ Successfully registered output directory: {output_dir}")


def write_env_sh(env_sh_lines):
    """
    Write the env.sh file
    """
    with open('.env.sh', 'w') as f:
        f.write('\n'.join(env_sh_lines))
        f.write('\n')

    print('*' * 80)
    print_green('Setup wizard completed successfully!')


def main():
    print('*' * 80)
    print('Running AlphaZeroArcade setup wizard...')
    print('*' * 80)

    os.chdir(os.path.dirname(__file__))

    try:
        env_sh_lines = []
        setup_output_dir(env_sh_lines)
        print('*' * 80)
        image = build_docker_image(env_sh_lines)
        print('*' * 80)
        validate_nvidia_installation(image)
        write_env_sh(env_sh_lines)
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
