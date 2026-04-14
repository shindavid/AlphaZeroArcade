#!/usr/bin/env python3
import os
import subprocess
import sys


def run(cmd: str, print_cmd=True, handler=None):
    if print_cmd:
        print(cmd)

    proc = subprocess.Popen(cmd, shell=True, text=True, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    if proc.returncode:
        print(stderr)
        if handler:
            handler(stderr)
        sys.exit(1)


extra_deps_dir = os.path.dirname(__file__)


def build_connect4():
    os.chdir(os.path.join(extra_deps_dir, 'connect4'))
    run('make')


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
        run('mkdir -p bin')
        os.chdir(os.path.join(extra_deps_dir, 'edax-reversi', 'src'))
        run(f'make build ARCH={arch} COMP={comp} OS={platform}')


def build_stockfish():
    stockfish_dir = os.path.join(extra_deps_dir, 'stockfish')
    binary = os.path.join(stockfish_dir, 'stockfish-ubuntu-x86-64-avx2')

    if os.path.exists(binary):
        print('stockfish binary already exists, skipping download')
        return

    os.makedirs(stockfish_dir, exist_ok=True)
    url = 'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar'
    tar_file = os.path.join(stockfish_dir, 'stockfish-ubuntu-x86-64-avx2.tar')

    run(f'curl -L -o {tar_file} {url}')
    run(f'tar xf {tar_file} -C {stockfish_dir} --strip-components=1')
    run(f'rm {tar_file}')


build_connect4()
build_edax()
build_stockfish()
