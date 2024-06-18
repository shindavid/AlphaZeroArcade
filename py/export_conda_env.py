#!/usr/bin/env python3
import os
from subprocess import run

from util.repo_util import Repo

export_env_cmd = "conda env export"
export_out = run(
    export_env_cmd.split(),
    capture_output=True, text=True, shell=False
)
lines = export_out.stdout.splitlines()

is_valid_format = len(lines) >= 2 \
    and lines[0].startswith('name:') \
    and lines[-1].startswith('prefix:')

if not is_valid_format:
    print(export_out.stdout)
    raise Exception("Environment file format is not supported.")

file_name = "environment.yml"
file_path = os.path.join(Repo.root(), file_name)

with open(file_path, 'w') as fp:
    fp.write('\n'.join(lines[1: -1] + [""]))

print(f"File {file_name} is saved to {file_path}.")
