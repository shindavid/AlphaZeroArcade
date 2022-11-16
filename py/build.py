#!/usr/bin/env python3

"""
TODO: incorporate ninja.
"""
import argparse
import os
import subprocess
import sys
from typing import List

from config import Config


def run(cmd: str):
    print(cmd)
    if os.system(cmd):
        sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--debug', action='store_true', help='debug build')
    parser.add_argument("-t", '--target', help='build targets, comma-separated. Default: all')
    parser.add_argument("-j", '--parallel', type=int,
                        help='make -j value (for build parallelism). Uses config value cmake.j if available. '
                             'Else uses cmake default')
    return parser.parse_args()


def get_bins(targets: List[str], args) -> List[str]:
    if targets:
        bin_postfix = 'd' if args.debug else ''
        return [f'{t}{bin_postfix}' for t in targets]
    try:
        cmd = 'cmake --build . --target help'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
        stdout = proc.communicate()[0]
        bins = []
        for line in stdout.splitlines():
            if line.startswith('... '):
                bins.append(line.split()[1])
        return bins
    except:
        return ['???']


def main():
    args = get_args()
    debug = bool(args.debug)

    torch_dir = Config.instance().get('libtorch_dir')
    j_value = args.parallel
    if not j_value:
        j_value = int(Config.instance().get('cmake.j', 0))

    targets = args.target.split(',') if args.target else []

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg = Config.instance().filename
    torch_link = 'https://pytorch.org/get-started/locally/'
    assert torch_dir, (f'Please download torch for C++ from {torch_link}, unzip to a path LIBTORCH_PATH, and then add an '
                       f'entry "libtorch_dir=LIBTORCH_PATH" to {cfg}. '
                       'An explicitly downloaded libtorch must be used instead of the one that ships with conda, '
                       'due to the cxx11-abi issue. See here: https://github.com/pytorch/pytorch/issues/17492')
    assert os.path.isdir(torch_dir)

    conda_prefix = os.environ.get('CONDA_PREFIX', None)
    assert conda_prefix, 'It appears you do not have a conda environment activated. Please activate!'
    eigen_dir = os.path.join(conda_prefix, 'share/eigen3/cmake')
    assert os.path.isdir(eigen_dir), 'Please conda install eigen.'

    lib_cmake_dir = os.path.join(conda_prefix, 'lib/cmake')
    assert os.path.isdir(lib_cmake_dir)
    found_boost = False
    for path in os.listdir(lib_cmake_dir):
        if path.startswith('Boost-') and os.path.isdir(os.path.join(lib_cmake_dir, path)):
            found_boost = True
            break
    assert found_boost, 'Please conda install boost.'

    os.chdir(repo_root)

    build_name = 'Debug' if debug else 'Release'
    target_dir = f'target/{build_name}'
    cmake_cmd_tokens = [
        'cmake',
        'CMakeLists.txt',
        f'-B{target_dir}',
        f'-DMY_TORCH_DIR={torch_dir}',
        f'-DCMAKE_PREFIX_PATH={conda_prefix}'
    ]
    if debug:
        cmake_cmd_tokens.append('-DCMAKE_BUILD_TYPE=Debug')
    cmake_cmd = ' '.join(cmake_cmd_tokens)
    run(cmake_cmd)

    os.chdir(target_dir)
    build_cmd_tokens = [
        'cmake',
        '--build',
        '.'
    ]
    for t in targets:
        build_cmd_tokens.extend(['--target', t])
    if j_value:
        build_cmd_tokens.append(f'-j{j_value}')

    build_cmd = ' '.join(build_cmd_tokens)
    run(build_cmd)

    bins = get_bins(targets, args)
    for b in bins:
        bin_loc = os.path.join(repo_root, target_dir, 'bin', b)
        if os.path.isfile(bin_loc):
            print(f'Binary location: {bin_loc}')


if __name__ == '__main__':
    main()
