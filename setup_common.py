# This file should not depend on any repo python files outside of the top-level directory.
#
# We keep this file at the top-level directory, rather than inside py/, to avoid PYTHONPATH-related
# import issues for scripts that are executed outside of a docker container, such as run_docker.py
# and setup_wizard.py.
#
# For scripts that executed inside of a docker container, we can be sure that the PYTHONPATH is
# correctly set up to include the py/ directory.

import json
import os
from packaging import version
from pathlib import Path
import subprocess


LOCAL_DOCKER_IMAGE = 'a0a'
DOCKER_HUB_IMAGE = 'dshin83/alphazeroarcade'
LATEST_DOCKER_HUB_IMAGE = f'{DOCKER_HUB_IMAGE}:latest'

# Note: on increases in the first component of the image-version, the pull_docker_image.py script
# automatically wipes the target/ directory.
MINIMUM_REQUIRED_IMAGE_VERSION = "14.1.1"

DIR = os.path.dirname(os.path.abspath(__file__))
ENV_JSON_FILENAME = os.path.join(DIR, '.env.json')

REQUIRED_PORTS = [
    5012,  # bokeh
    5013,  # bokeh2 - in case you want to launch two bokeh servers
    5173,  # vite
    52528,  # web bridge
    8002,  # flask
    8003,  # flask2 - in case you want to launch two flask servers
    8051,  # dash
    8888,  # jupyter-notebook
]


def update_env_json(mappings):
    env = {}
    if os.path.exists(ENV_JSON_FILENAME):
        with open(ENV_JSON_FILENAME) as f:
            env = json.load(f)
    env.update(mappings)
    with open(ENV_JSON_FILENAME, 'w') as f:
        json.dump(env, f, indent=2)


def get_env_json():
    env = {}
    if os.path.exists(ENV_JSON_FILENAME):
        with open(ENV_JSON_FILENAME) as f:
            env = json.load(f)
    return env


def get_image_label(image_name, label_key):
    """
    Get the value of a specific label from a Docker image.
    """
    result = subprocess.check_output(
        ["docker", "inspect",
            f"--format={{{{index .Config.Labels \"{label_key}\"}}}}", image_name],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()
    if not result:
        return None
    return result


def is_version_ok(version_str):
    """
    Check if a given version string is at least the minimum required version.
    """
    if version_str is None:
        return False
    return version.parse(version_str) >= version.parse(MINIMUM_REQUIRED_IMAGE_VERSION)


def is_subpath(child_path, parent_path):
    p, d = Path(child_path).resolve(), Path(parent_path).resolve()
    return p.is_relative_to(d)
