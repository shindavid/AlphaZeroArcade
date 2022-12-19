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
    parser.add_argument("-c", '--clear-core-dumps', action='store_true', help='rm core.* (in cwd) before doing anything')
    parser.add_argument("-t", '--target', help='build targets, comma-separated. Default: all')
    parser.add_argument("-j", '--parallel', type=int,
                        help='make -j value (for build parallelism). Uses config value cmake.j if available. '
                             'Else uses cmake default')
    parser.add_argument('-D', '--macro-defines', action='append',
                        help='macro definitions to forward to make cmd (-D FOO -D BAR=2). If a macro name is passed'
                        ' without an assigned value, it is given a value of "1" by default. This plays nicely with the'
                        ' IS_MACRO_ASSIGNED_TO_1() macro function defined in cpp/util/CppUtil.hpp')
    return parser.parse_args()


def get_targets(targets: List[str], args) -> List[str]:
    if targets:
        return targets
    try:
        cmd = 'cmake --build . --target help'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
        stdout = proc.communicate()[0]
        targets = []
        for line in stdout.splitlines():
            if line.startswith('... '):
                targets.append(line.split()[1])
        return targets
    except:
        return ['???']


def get_torch_dir():
    cfg = Config.instance().filename

    torch_dir = Config.instance().get('libtorch_dir')
    torch_link = 'https://pytorch.org/get-started/locally/'
    assert torch_dir, (f'Please download torch for C++ from {torch_link}, unzip to a path LIBTORCH_PATH, and then add an '
                       f'entry "libtorch_dir=LIBTORCH_PATH" to {cfg}. '
                       'An explicitly downloaded libtorch must be used instead of the one that ships with conda, '
                       'due to the cxx11-abi issue. See here: https://github.com/pytorch/pytorch/issues/17492')
    assert os.path.isdir(torch_dir)
    return torch_dir


def get_eigenrand_dir():
    cfg = Config.instance().filename

    eigenrand_dir = Config.instance().get('eigenrand_dir')
    eigenrand_link = 'https://github.com/bab2min/EigenRand.git'
    assert eigenrand_dir, (f'Please git clone {eigenrand_link} to EIGENRAND_PATH and add an '
                           f'entry "eigenrand_dir=EIGENRAND_PATH" to {cfg}.')
    assert os.path.isdir(eigenrand_dir)
    return eigenrand_dir


def get_conda_prefix():
    conda_prefix = os.environ.get('CONDA_PREFIX', None)
    assert conda_prefix, 'It appears you do not have a conda environment activated. Please activate!'
    return conda_prefix


def check_for_eigen_dir(conda_prefix):
    eigen_dir = os.path.join(conda_prefix, 'share/eigen3/cmake')
    assert os.path.isdir(eigen_dir), 'Please conda install eigen.'


def check_for_boost_dir(conda_prefix):
    lib_cmake_dir = os.path.join(conda_prefix, 'lib/cmake')
    assert os.path.isdir(lib_cmake_dir)
    for path in os.listdir(lib_cmake_dir):
        if path.startswith('Boost-') and os.path.isdir(os.path.join(lib_cmake_dir, path)):
            return
    raise Exception('Please conda install boost.')


def main():
    args = get_args()

    if args.clear_core_dumps:
        run('rm -f core.*')

    debug = bool(args.debug)

    j_value = args.parallel
    if not j_value:
        j_value = int(Config.instance().get('cmake.j', 0))

    targets = args.target.split(',') if args.target else []
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(repo_root)

    torch_dir = get_torch_dir()
    eigenrand_dir = get_eigenrand_dir()
    conda_prefix = get_conda_prefix()
    check_for_eigen_dir(conda_prefix)
    check_for_boost_dir(conda_prefix)

    macro_defines = args.macro_defines if args.macro_defines else []
    macro_defines = [f'{d}=1' if d.find('=') == -1 else d for d in macro_defines]
    extra_definitions = ' '.join(f'-D{d}' for d in macro_defines)

    build_name = 'Debug' if debug else 'Release'
    target_dir = f'target/{build_name}'
    cmake_cmd_tokens = [
        'cmake',
        'CMakeLists.txt',
        f'-B{target_dir}',
        f'-DMY_TORCH_DIR={torch_dir}',
        f'-DMY_EIGENRAND_DIR={eigenrand_dir}',
        f'-DCMAKE_PREFIX_PATH={conda_prefix}',
        f'-DEXTRA_DEFINITIONS="{extra_definitions}"',
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

    bin_postfix = 'd' if args.debug else ''
    bins = get_targets(targets, args)
    for b in bins:
        bin_loc = os.path.join(repo_root, target_dir, 'bin', f'{b}{bin_postfix}')
        if os.path.isfile(bin_loc):
            print(f'Binary location: {bin_loc}')


if __name__ == '__main__':
    main()

