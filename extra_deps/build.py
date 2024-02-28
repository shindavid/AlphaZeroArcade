#!/usr/bin/env python3
import os
import subprocess


extra_deps_dir = os.path.dirname(__file__)


def build_connect4():
    os.chdir(os.path.join(extra_deps_dir, 'connect4'))
    subprocess.run('make', shell=True)


def build_edax():
    arch = 'x64-modern'
    comp = 'gcc'
    platform = 'linux'

    platform_short = platform[0]
    binary = f'{platform_short}Edax-{arch}'
    full_binary = os.path.join(extra_deps_dir, 'edax-reversi', 'bin', binary)

    # edax-reversi Makefile doesn't skip build if binary exists, so we check for binary manually
    if os.path.exists(full_binary):
        print(f'{binary} already exists, skipping edax-reversi build')
    else:
        os.chdir(os.path.join(extra_deps_dir, 'edax-reversi'))
        subprocess.run('mkdir -p bin', shell=True)
        os.chdir(os.path.join(extra_deps_dir, 'edax-reversi', 'src'))
        subprocess.run(f'make build ARCH={arch} COMP={comp} OS={platform}', shell=True)


build_connect4()
build_edax()
