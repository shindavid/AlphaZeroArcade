# This file should not depend on any repo python files outside of the top-level directory.

import json
import os


LOCAL_DOCKER_IMAGE = 'a0a'
DOCKER_HUB_IMAGE = 'dshin83/alphazeroarcade:latest'

DIR = os.path.dirname(os.path.abspath(__file__))
ENV_JSON_FILENAME = os.path.join(DIR, '.env.json')


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