#!/usr/bin/env python3
import argparse
import os
import pkg_resources
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
    parser.add_argument('--clean', action='store_true', help='clean out target/.../bin/ directory')
    parser.add_argument("-c", '--clear-core-dumps', action='store_true', help='rm core.* (in cwd) before doing anything')
    parser.add_argument("-t", '--target', help='build targets, comma-separated. Default: all')
    parser.add_argument('-D', '--macro-defines', action='append',
                        help='macro definitions to forward to make cmd (-D FOO -D BAR=2). If a macro name is passed'
                        ' without an assigned value, it is given a value of "1" by default. This plays nicely with the'
                        ' IS_MACRO_ENABLED() macro function defined in cpp/util/CppUtil.hpp')
    return parser.parse_args()


def validate_gcc_version():
    """
    Our c++ code uses std::atomic<std::shared_ptr>>, which is only supported in gcc-12+.

    See: https://en.cppreference.com/w/cpp/compiler_support#C.2B.2B20_library_features
    """
    output = subprocess.getoutput('gcc --version')
    version_str = output.splitlines()[0].split()[-1]
    version = pkg_resources.parse_version(version_str)
    required_version_str = '12'
    required_version = pkg_resources.parse_version(required_version_str)
    if version >= required_version:
        return

    print(f'Your gcc version ({version_str}) is old. Please update to version {required_version_str}+')
    print('Recommended action:')
    print('')
    print('sudo apt-get install gcc-12')
    print('sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12')
    sys.exit(0)


def validate_gxx_version():
    """
    Our c++ code uses std::atomic<std::shared_ptr>>, which is only supported in gcc-12+.

    See: https://en.cppreference.com/w/cpp/compiler_support#C.2B.2B20_library_features
    """
    output = subprocess.getoutput('g++ --version')
    version_str = output.splitlines()[0].split()[-1]
    version = pkg_resources.parse_version(version_str)
    required_version_str = '12'
    required_version = pkg_resources.parse_version(required_version_str)
    if version >= required_version:
        return

    print(f'Your g++ version ({version_str}) is old. Please update to version {required_version_str}+')
    print('Recommended action:')
    print('')
    print('sudo apt install g++-12')
    print('sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12')
    sys.exit(0)


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


def get_conda_prefix():
    # detect if there is an active anaconda env, and if so, return a path to it
    return os.environ.get('CONDA_PREFIX', None)


def main():
    cwd = os.getcwd()
    args = get_args()

    if args.clear_core_dumps:
        run('rm -f core.*')

    validate_gcc_version()
    validate_gxx_version()

    debug = bool(args.debug)

    targets = args.target.split(',') if args.target else []
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(repo_root)

    torch_dir = get_torch_dir()
    eigenrand_dir = os.path.join(repo_root, 'extra_deps/EigenRand')
    tinyexpr_dir = os.path.join(repo_root, 'extra_deps/tinyexpr')

    macro_defines = args.macro_defines if args.macro_defines else []
    macro_defines = [f'{d}=1' if d.find('=') == -1 else d for d in macro_defines]
    extra_definitions = ' '.join(f'-D{d}' for d in macro_defines)

    build_name = 'Debug' if debug else 'Release'
    target_dir = f'target/{build_name}'

    if args.clean:
        run(f'rm -rf {target_dir}/bin/*')

    cmake_cmd_tokens = [
        'cmake',
        '-G Ninja',
        'CMakeLists.txt',
        f'-B{target_dir}',
        f'-DMY_TORCH_DIR={torch_dir}',
        f'-DMY_EIGENRAND_DIR={eigenrand_dir}',
        f'-DMY_TINYEXPR_DIR={tinyexpr_dir}',
        f'-DEXTRA_DEFINITIONS="{extra_definitions}"',
    ]
    conda_prefix = get_conda_prefix()
    if conda_prefix:
        cmake_cmd_tokens.append(f'-DCONDA_PREFIX={conda_prefix}')
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

    build_cmd = ' '.join(build_cmd_tokens)
    run(build_cmd)

    bin_postfix = 'd' if args.debug else ''
    bins = get_targets(targets, args)
    for b in bins:
        bin_loc = os.path.join(repo_root, target_dir, 'bin', f'{b}{bin_postfix}')
        if os.path.isfile(bin_loc):
            relative_bin_loc = os.path.relpath(bin_loc, cwd)
            print(f'Binary location: {relative_bin_loc}')


if __name__ == '__main__':
    main()
