#!/usr/bin/env python3
import os

# Remove the first line which contains the environments name
# Remove the last line which contains the prefix
cmd = 'conda env export | head -n -1 | tail -n +2 > environment.yml'

print('Running from repo root:\n')
print(cmd)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(repo_root)

os.system(cmd)

