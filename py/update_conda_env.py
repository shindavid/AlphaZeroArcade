#!/usr/bin/env python3
import os

cmd = 'conda env update --file environment.yml --prune'

print('Running from repo root:\n')
print(cmd)

print('')
print('This may irreversibly change your conda env.')
response = input('Are you sure you want to continue? [y/N]: ')
if response in ('y', 'Y'):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(repo_root)

    os.system(cmd)
else:
    print('\nExiting without update.')

