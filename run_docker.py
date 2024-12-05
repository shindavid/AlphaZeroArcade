#!/usr/bin/env python3
from setup_wizard import MD5_FILE_PATHS

import argparse
import hashlib
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.resolve()

EXPOSED_PORTS = [
    5012,  # bokeh
    8002,  # flask
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--skip-hash-check', action='store_true', help='skip hash check')
    parser.add_argument("-i", '--instance-name', default='a0a_instance',
                        help='name of the instance to run (default: %(default)s)')
    return parser.parse_args()


def check_hashes(env_vars):
    for path in MD5_FILE_PATHS:
        full_path = Path(REPO_ROOT) / path
        if not full_path.exists():
            raise Exception(f"Error: {full_path} not found.")

        env_key = f'A0A_MD5_{path}'
        expected_hash = env_vars.get(env_key)
        if not expected_hash:
            return False

        with full_path.open("rb") as f:
            actual_hash = hashlib.md5(f.read()).hexdigest()

        if actual_hash != expected_hash:
            return False

    return True


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
    docker_cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)

    try:
        print(f"Executing into Docker container: {docker_cmd_str}")
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing into Docker container: {e}")


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

    if not args.skip_hash_check:
        if not check_hashes(env_vars):
            print('Your docker image appears out of date.')
            print('Please run setup_wizard.py again.')
            return None

    return env_vars


def run_container(args):
    env_vars = get_env_vars(args)
    if env_vars is None:
        return

    # Get variables from the environment
    A0A_OUTPUT_DIR = env_vars.get("A0A_OUTPUT_DIR")
    A0A_DOCKER_IMAGE = env_vars.get("A0A_DOCKER_IMAGE")

    if not A0A_OUTPUT_DIR or not A0A_DOCKER_IMAGE:
        print("Error: A0A_OUTPUT_DIR or A0A_DOCKER_IMAGE not set in .env.sh")
        return

    output_dir = Path(A0A_OUTPUT_DIR).resolve()

    # Check if output_dir is inside REPO_ROOT
    if output_dir.resolve().is_relative_to(REPO_ROOT.resolve()):
        # Handle overlapping mount points
        relative_output = output_dir.relative_to(REPO_ROOT)
        mounts = ['-v', f"{REPO_ROOT}:/workspace"]
        symlink_cmd = f"ln -sf /workspace/{relative_output} /output"
        post_mount_cmds = [symlink_cmd]
    else:
        # Separate mounts for REPO_ROOT and output_dir
        mounts = ['-v', f"{REPO_ROOT}:/workspace", '-v', f"{output_dir}:/output"]
        post_mount_cmds = []

    ports_strs = []
    for port in EXPOSED_PORTS:
        ports_strs += ['-p', f"{port}:{port}"]

    # Build the docker run command
    docker_cmd = [
        "docker", "run", "--rm", "-it", "--gpus", "all", "--name", args.instance_name,
    ] + ports_strs + mounts + [
        A0A_DOCKER_IMAGE
    ]

    # Add post-mount commands as the container's entrypoint
    if post_mount_cmds:
        entrypoint_cmd = " && ".join(post_mount_cmds)
        entrypoint_cmd += "; exec bash"
        docker_cmd += ["bash", "-c", entrypoint_cmd]

    docker_cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)

    # Run the docker command
    try:
        print(f"Running Docker command: {docker_cmd_str}")
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")


def main():
    args = get_args()

    if is_container_running(args.instance_name):
        execute_into_container(args.instance_name)
    else:
        run_container(args)


if __name__ == "__main__":
    main()
