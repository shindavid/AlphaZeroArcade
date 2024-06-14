#!/usr/bin/env python3
import os
from subprocess import run

from util.repo_util import Repo

export_env_cmd = "conda env export"
export_out = run(
    export_env_cmd.split(),
    capture_output=True, text=True, shell=False
)
# Ensure the diff is as expected
# first and last key should be name and prefix respectively
lines = export_out.stdout.splitlines()
name_key = "name:"
prefix_key = "prefix:"

invalid = not len(lines) >= 2 \
    and lines[0].startswith(name_key) \
    and lines[-1].startswith(prefix_key)

if invalid:
    print(export_out.stdout)
    raise Exception("Environment file format is not supported.")

file_path = os.path.join(Repo.root(), "environment.yml")

with open(file_path, 'w') as fp:
    # New lines are handled correctly in Python 3
    fp.write('\n'.join(lines[1: -1]))

print(f"File environment.yml is saved to {file_path}.")
