#!/usr/bin/env python3
import json
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
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def validate_conda_env_name(env_name):
    """
    Validate that the conda environment name is valid (alphanumeric + underscores only)
    """
    for c in env_name:
        if c.isalnum() or c == '_':
            continue
        raise SetupException(f'Invalid conda environment name: {env_name}')


def get_conda_envs():
    cmd = 'conda info -e --json'
    proc = subprocess.run(cmd, shell=True, capture_output=True, encoding='utf-8')
    if proc.returncode:
        print('*' * 80)
        print('Failed to get conda environments')
        print(f'cmd: {cmd}')
        print('stdout:')
        print(proc.stdout)
        print('stderr:')
        print(proc.stderr)
        raise SetupException()

    envs = []
    for env in json.loads(proc.stdout)['envs']:
        parts = env.split('/')
        if len(parts) > 2 and parts[-2] == 'envs':
            envs.append(parts[-1])

    return envs


def setup_conda(env_sh_lines):
    """
    Checks first that conda is installed.

    Request user to select a conda environment name, using "AlphaZeroArcade" as the default.
    If the environment does not exist, create it
    """

    # Check that conda is installed:
    proc = subprocess.run(['conda', '--version'], capture_output=True)
    if proc.returncode:
        raise SetupException('Conda is not installed. Please install it first.')

    conda_envs = get_conda_envs()

    print('Please select a conda environment to use for this project.')
    print('If the environment does not exist, it will be created with environment.yml.')
    print(f'Existing conda environments: {", ".join(conda_envs)}')
    print('')

    default_env_name = os.environ.get('A0A_CONDA_ENV', 'AlphaZeroArcade')
    env_name = input(f'Enter conda environment name [{default_env_name}]: ')
    if not env_name:
        env_name = default_env_name
    validate_conda_env_name(env_name)
    env_sh_lines.append(f'export A0A_CONDA_ENV={env_name}')
    env_sh_lines.append(f'conda activate $A0A_CONDA_ENV')

    # check whether the environment exists
    if env_name not in conda_envs:
        print(f'Creating conda environment: {env_name}')
        if os.system(f'conda env create -f environment.yml -n {env_name}'):
            raise VerboseSetupException('Failed to create conda environment.')
        print(f'Conda environment {env_name} successfully created!')
    else:
        print(f'Conda environment {env_name} already exists, skipping creation.')
    return env_name


def cuda_check():
    proc = subprocess.run(['nvcc', '--version'], capture_output=True)
    if proc.returncode:
        print_red('CUDA is not detected on your system, but is required for this project.')
        print('')
        print('Please visit the NVIDIA CUDA Toolkit download page to download and install it:')
        print('')
        print('   https://developer.nvidia.com/cuda-downloads')
        print('')
        print("Follow NVIDIA's installation instructions to ensure proper setup.")
        print('After installation, please rerun this setup script.')
        print('')
        print_red(
            'Note: installing CUDA can be a complex process, especially on laptops.')
        print_red(
            'Unfortunately, this part cannot be automated due to licensing restrictions.')
        raise SetupException()


def setup_libtorch(env_sh_lines):
    """
    Request user to set libtorch directory. Validate their input by checking that the directory
    exists, and that it contains some expected files.
    """
    print('*' * 80)
    print('This project requires a libtorch installation.')
    print('If you do not have libtorch installed, please visit the following link to')
    print('download it:')
    print('')
    print('   https://pytorch.org/get-started/locally/')
    print('')
    print_green('Please make sure to select Package: LibTorch, and to select a ComputePlatform')
    print_green('that matches your installed CUDA version.')
    print('')
    print_green(
        'Also, make sure to download the cxx11 ABI version, NOT the Pre-cxx11 ABI')
    print_green('version.')
    print('')

    default_location = os.environ.get('A0A_LIBTORCH_DIR', None)
    if default_location is None:
        prompt = 'Please enter the location of your libtorch installation: '
    else:
        prompt = f'Please enter the location of your libtorch installation [{default_location}]: '
    while True:
        location = input(prompt).strip()
        if not location:
            location = default_location
            if not location:
                continue
        expanded_location = os.path.expanduser(location)
        if not os.path.isdir(expanded_location):
            print_red(f'Directory {location} does not exist.')
            continue

        # Check for some expected files
        filename = os.path.join(expanded_location, 'lib', 'libtorch.so')
        if not os.path.isfile(filename):
            print_red(
                f'Directory {location} does not appear to be a valid libtorch installation,')
            print_red(f'since file {filename} does not exist.')
            continue

        # TODO: automated check that it is cxx11 ABI
        break

    env_sh_lines.append(f'export A0A_LIBTORCH_DIR={expanded_location}')


def setup_output_dir(env_sh_lines):
    """
    Request user to set the output directory.
    """
    print('*' * 80)
    print('Alphazero runs write (a lot of) data to disk. Please specify a directory where')
    print('that data will be written. If you have a fast SSD, it is recommended to use that')
    print('for the data directory.')
    print('')
    print('If you are only using this checkout to run a self-play or ratings server, and not ')
    print('for a loop-controller, then your choice here does not matter.')

    cwd = os.getcwd()
    default_output_dir = os.environ.get('A0A_OUTPUT_DIR', os.path.join(cwd, 'output'))
    prompt = f'Please enter the location of your output directory [{default_output_dir}]: '
    output_dir = input(prompt).strip()
    if not output_dir:
        output_dir = default_output_dir

    expanded_output_dir = os.path.expanduser(output_dir)
    env_sh_lines.append(f'export A0A_OUTPUT_DIR={expanded_output_dir}')


def setup_local_python_imports(conda_env):
    if os.system(f'cd py && python setup.py develop'):
        raise SetupException('Failed setup.py')


def build_extra_deps():
    if os.system(f'cd extra_deps && python build.py'):
        raise SetupException('Failed to build extra_deps')


def write_env_sh(env_sh_lines):
    """
    Write the env.sh file
    """
    with open('.env.sh', 'w') as f:
        f.write('\n'.join(env_sh_lines))
        f.write('\n')

    print('*' * 80)
    print_green('Setup wizard completed successfully!')
    print('')
    print('Please make sure to run the following whenever you start a new shell:')
    print('')
    print('    source env_setup.sh')


def main():
    print('*' * 80)
    print('Running AlphaZeroArcade setup wizard...')
    print('*' * 80)

    os.chdir(os.path.dirname(__file__))

    try:
        env_sh_lines = []
        cuda_check()
        conda_env = setup_conda(env_sh_lines)
        setup_libtorch(env_sh_lines)
        setup_output_dir(env_sh_lines)
        setup_local_python_imports(conda_env)
        build_extra_deps()
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
