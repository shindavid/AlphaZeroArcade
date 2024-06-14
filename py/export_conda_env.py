#!/usr/bin/env python3
import os
from subprocess import run


export_env_cmd = "conda env export"
export_out = run(
    export_env_cmd,
    capture_output=True, text=True, shell=True
)
# Ensure the diff is as expected
# first and last key should be name and prefix respectively
lines = export_out.stdout.splitlines()
name_key = 'name:'
prefix_key = 'prefix:'
assert \
    len(lines) >= 2 \
    and lines[0][: len(name_key)] == name_key  \
    and lines[-1][: len(prefix_key)] == prefix_key \
    , "Environment file format is not supported."


file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "environment.yml"
)

with open(file_path, "w") as fp:
    # New lines are handled correctly in Python 3
    fp.write('\n'.join(lines[1: -1]))
print('File environment.yml is saved to root repo.')