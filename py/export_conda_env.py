#!/usr/bin/env python3
import os

cmd = 'conda env export > environment.yml'

print('Running from repo root:\n')
print(cmd)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(repo_root)

os.system(cmd)

