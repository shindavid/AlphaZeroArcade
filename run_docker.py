#!/usr/bin/env python3

# This file should not depend on any repo python files outside of the top-level directory.

from setup_common import MINIMUM_REQUIRED_IMAGE_VERSION, get_env_json, get_image_label, \
    is_version_ok

import argparse
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.resolve()

EXPOSED_PORTS = [
    5012,  # bokeh
    8002,  # flask
    8051,  # dash
]


def check_image_version(image_name):
    min_version = MINIMUM_REQUIRED_IMAGE_VERSION
    image_version = get_image_label(image_name, 'version')

    # Check if the image version is at least MINIMUM_REQUIRED_IMAGE_VERSION

    if not is_version_ok(image_version):
        if image_version is None:
            print('Your docker image appears out of date.')
        else:
            print(f'Your docker image version is {image_version}, but the minimum required version is {min_version}.')
        print('')
        print('Please refresh your docker image by running pull_docker_image.py.')
        print('')
        print('Or, to run anyways, rerun with --skip-image-version-check')
        return False

    return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--skip-image-version-check', action='store_true',
                        help='skip image version check')
    parser.add_argument("-d", '--docker-image',
                        help='name of the docker image to use (default: comes from .env.json)')
    parser.add_argument("-i", '--instance-name', default='a0a_instance',
                        help='name of the instance to run (default: %(default)s)')
    return parser.parse_args()


def is_container_running(container_name):
    cmd = [
        "docker", "inspect",
        "--format={{.State.Running}}",
        container_name
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if result.returncode == 0:
        output = result.stdout.strip().lower()
        return output == 'true'
    else:
        # Container does not exist or an error occurred
        return False


def execute_into_container(container_name):
    docker_cmd = ["docker", "exec", "-it", container_name, "bash"]
    launch_docker_cmd(docker_cmd, run=False)


def get_env_vars(args):
    env_file = REPO_ROOT / ".env.sh"

    if not env_file.exists():
        print(f"Error: {env_file} not found. Run setup_wizard.py first.")
        return None

    env_vars = {}
    with env_file.open() as f:
        for line in f:
            if line.startswith("export"):
                key, value = line.replace("export ", "").strip().split("=", 1)
                env_vars[key] = value.strip()

    return env_vars


def run_container(args):
    env = get_env_json()

    output_dir = env.get("OUTPUT_DIR", None)
    docker_image = args.docker_image
    if not docker_image:
        docker_image = env.get("DOCKER_IMAGE", None)

    if not output_dir or not docker_image:
        print("Error: Bad environment. Please run setup_wizard.py first.")
        return

    if not args.skip_image_version_check:
        if not check_image_version(docker_image):
            return

    libtorch_dir = REPO_ROOT / "libtorch"
    libtorch_dir.mkdir(exist_ok=True)

    output_dir = Path(output_dir)
    mounts = ['-v', f"{REPO_ROOT}:/workspace/repo", '-v', f"{libtorch_dir}:/workspace/libtorch"]
    post_mount_cmds = ["export PYTHONPATH=/workspace/repo/py"]

    # Check if output_dir is inside REPO_ROOT
    if output_dir.resolve().is_relative_to(REPO_ROOT.resolve()):
        # Handle overlapping mount points
        relative_output = output_dir.relative_to(REPO_ROOT)
        symlink_cmd = f"ln -sf /workspace/repo/{relative_output} /workspace/output"
        post_mount_cmds.extend([symlink_cmd])
    else:
        # Separate mounts for REPO_ROOT and output_dir
        mounts.extend(['-v', f"{output_dir}:/workspace/output"])

    ports_strs = []
    for port in EXPOSED_PORTS:
        ports_strs += ['-p', f"{port}:{port}"]

    user_id = subprocess.check_output(["id", "-u"], text=True).strip()
    group_id = subprocess.check_output(["id", "-g"], text=True).strip()

    if (user_id, group_id) != ('1000', '1000'):
        # TODO: add support for non-1000 users. Will require some surgery to the Dockerfile and
        # entrypoint.sh. Need to test on both local machine and on runpod.
        #
        # For now, just print the below warning.

        user = subprocess.check_output(["id", "-un"], text=True).strip()
        group = subprocess.check_output(["id", "-gn"], text=True).strip()

        CRED = '\033[91m'
        CEND = '\033[0m'
        print(CRED, end='')

        print('#' * 70)
        print('# %-66s #' % 'WARNING!')
        if user_id != '1000':
            print('# %-66s #' % f'Your user id ({user}:{user_id}) is not 1000.')
        if group_id != '1000':
            print('# %-66s #' % f'The group id ({group}:{group_id}) is not 1000.')

        'As a result, outside of the docker container, you might need sudo permissions'
        print('# %-66s #' % 'As a result, outside of the docker container, you might need sudo')
        print('# %-66s #' % 'permissions to delete/modify files/dirs written inside this docker')
        print('# %-66s #' % 'container. This includes directories like output/ and target/.')
        print('#' * 70)
        print(CEND, end='')

    # Build the docker run command
    docker_cmd = [
        "docker", "run", "--rm", "-it", "--gpus", "all", "--name", args.instance_name,
        '--user', '1000:1000',
    ] + ports_strs + mounts + [
        docker_image
    ]

    # Add post-mount commands as the container's entrypoint
    if post_mount_cmds:
        entrypoint_cmd = " && ".join(post_mount_cmds)
        entrypoint_cmd += "; exec bash"
        docker_cmd += ["bash", "-c", entrypoint_cmd]

    launch_docker_cmd(docker_cmd, run=True)


def launch_docker_cmd(docker_cmd, run: bool):
    if run:
        msg = 'Running Docker container'
        error_msg = 'Error running Docker container'
    else:
        msg = 'Executing into Docker container'
        error_msg = 'Error executing into Docker container'

    # Determine if we're in a tmux session.
    in_tmux = "TMUX" in os.environ

    # If yes, read the current window name and rename to "docker"
    old_name = None
    if in_tmux:
        old_name = subprocess.check_output(
            ["tmux", "display-message", "-p", "#W"],
            text=True).strip()

        subprocess.run(["tmux", "rename-window", "docker"], check=True)

    docker_cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)

    try:
        print(f"{msg}: {docker_cmd_str}")
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{error_msg}: {e}")
    finally:
        # If we renamed the window, revert it now
        if old_name is not None:
            subprocess.run(["tmux", "rename-window", old_name], check=True)


def main():
    args = get_args()

    if is_container_running(args.instance_name):
        execute_into_container(args.instance_name)
    else:
        run_container(args)


if __name__ == "__main__":
    main()
