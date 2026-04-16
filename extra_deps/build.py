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


STOCKFISH_URL = 'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar'
STOCKFISH_FILE = STOCKFISH_URL.rsplit('/', 1)[-1]

def build_stockfish():
    stockfish_dir = os.path.join(extra_deps_dir, 'stockfish')
    binary = os.path.join(stockfish_dir, 'stockfish-ubuntu-x86-64-avx2')

    if os.path.exists(binary):
        print('stockfish binary already exists, skipping download')
        return

    os.makedirs(stockfish_dir, exist_ok=True)
    tar_file = os.path.join(stockfish_dir, STOCKFISH_FILE)

    run(f'curl -L -o {tar_file} {STOCKFISH_URL}')
    run(f'tar xf {tar_file} -C {stockfish_dir} --strip-components=1')
    run(f'rm {tar_file}')


LC0_NETWORK_URL = 'https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz'
LC0_NETWORK_FILE = LC0_NETWORK_URL.rsplit('/', 1)[-1]


def build_lc0():
    lc0_dir = os.path.join(extra_deps_dir, 'lc0')
    binary = os.path.join(lc0_dir, 'lc0')
    network = os.path.join(lc0_dir, LC0_NETWORK_FILE)

    if not os.path.exists(binary):
        src_dir = os.path.join(extra_deps_dir, 'lc0-src')

        if not os.path.exists(src_dir):
            run(f'git clone -b release/0.32 --depth 1 https://github.com/LeelaChessZero/lc0.git {src_dir}')

        os.chdir(src_dir)
        run(f'PATH="$HOME/.local/bin:$PATH" ./build.sh')

        os.makedirs(lc0_dir, exist_ok=True)
        run(f'cp {src_dir}/build/release/lc0 {binary}')
        run(f'rm -rf {src_dir}')
    else:
        print('lc0 binary already exists, skipping build')

    if not os.path.exists(network):
        os.makedirs(lc0_dir, exist_ok=True)
        download_cmd = f'curl -fL --retry 3 --connect-timeout 30 -o {network} {LC0_NETWORK_URL}'
        print(download_cmd)
        proc = subprocess.Popen(download_cmd, shell=True, text=True, stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        if proc.returncode:
            # Clean up partial/error file
            if os.path.exists(network):
                os.remove(network)
            print(f'Warning: failed to download lc0 network (server may be down).')
            print(f'You can download it manually later:')
            print(f'  curl -fL -o {network} {LC0_NETWORK_URL}')
    else:
        print(f'{LC0_NETWORK_FILE} already exists, skipping download')


build_connect4()
build_edax()
build_stockfish()
build_lc0()
