#!/usr/bin/env python3
import subprocess
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.resolve()

    # Load environment variables from .env.sh
    env_file = repo_root / ".env.sh"
    if not env_file.exists():
        print(f"Error: {env_file} not found. Run setup_wizard.py first.")
        return

    # Source the .env.sh file
    env_vars = {}
    with env_file.open() as f:
        for line in f:
            if line.startswith("export"):
                key, value = line.replace("export ", "").strip().split("=", 1)
                env_vars[key] = value.strip()

    # Get variables from the environment
    A0A_OUTPUT_DIR = env_vars.get("A0A_OUTPUT_DIR")
    A0A_DOCKER_IMAGE = env_vars.get("A0A_DOCKER_IMAGE")

    if not A0A_OUTPUT_DIR or not A0A_DOCKER_IMAGE:
        print("Error: A0A_OUTPUT_DIR or A0A_DOCKER_IMAGE not set in .env.sh")
        return

    output_dir = Path(A0A_OUTPUT_DIR).resolve()

    # Check if output_dir is inside repo_root
    if output_dir.resolve().is_relative_to(repo_root.resolve()):
        # Handle overlapping mount points
        relative_output = output_dir.relative_to(repo_root)
        mounts = ['-v', f"{repo_root}:/workspace"]
        symlink_cmd = f"ln -sf /workspace/{relative_output} /output"
        post_mount_cmds = [symlink_cmd]
    else:
        # Separate mounts for repo_root and output_dir
        mounts = ['-v', f"{repo_root}:/workspace", '-v', f"{output_dir}:/output"]
        post_mount_cmds = []

    EXPOSED_PORTS = [
        5012,  # bokeh
        8002,  # flask
        ]
    ports_strs = []
    for port in EXPOSED_PORTS:
        ports_strs += ['-p', f"{port}:{port}"]

    # Build the docker run command
    docker_cmd = [
        "docker", "run", "--rm", "-it", "--gpus", "all",
    ] + ports_strs + mounts + [
        A0A_DOCKER_IMAGE
    ]

    # Run the docker command
    try:
        # Add post-mount commands as the container's entrypoint
        if post_mount_cmds:
            entrypoint_cmd = " && ".join(post_mount_cmds)
            entrypoint_cmd += "; exec bash"
            docker_cmd += ["bash", "-c", entrypoint_cmd]

        print(f"Running Docker command: {' '.join(docker_cmd)}")
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")


if __name__ == "__main__":
    main()
