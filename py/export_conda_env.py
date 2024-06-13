#!/usr/bin/env python3
import os
from subprocess import run


export_env_cmd = "conda env export"
# Remove the first line which contains the environments name
# Remove the last line which contains the prefix
export_stripped_env_cmd = f"{export_env_cmd} | head -n -1 | tail -n +2"
# Ensure the diff is as expected
export_out = run(
    export_env_cmd,
    capture_output=True, text=True, shell=True
)
lines = export_out.stdout.split()
assert \
    len(lines) >= 2 \
    and lines[0] == 'name:'  \
    and lines[-2] == 'prefix:', \
    "environment file format is not supported"


cmd = f'{export_stripped_env_cmd} > environment.yml'
print('Running from repo root:\n')
print(cmd)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(repo_root)

os.system(cmd)

