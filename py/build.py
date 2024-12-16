#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import os
import pkg_resources
import subprocess
import sys
from typing import List

from games.index import GAME_SPECS_BY_NAME
from util.py_util import CustomHelpFormatter


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


def get_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument("-d", '--debug', action='store_true', help='debug build')
    parser.add_argument('--clean', action='store_true', help='clean out target/.../bin/ directory')
    parser.add_argument("-c", '--clear-core-dumps', action='store_true', help='rm core.* (in cwd) before doing anything')
    parser.add_argument("-t", '--target', help='build targets, comma-separated. Default: all')
    parser.add_argument('-D', '--macro-defines', action='append',
                        help='macro definitions to forward to make cmd (-D FOO -D BAR=2). If a macro name is passed'
                        ' without an assigned value, it is given a value of "1" by default. This plays nicely with the'
                        ' IS_MACRO_ENABLED() macro function defined in cpp/util/CppUtil.hpp')
    return parser.parse_args()


def catch_first_time_ninja_error(stderr):
    lines = stderr.splitlines()
    if lines[0].startswith('CMake Error: Error: generator : Ninja'):
        print('\033[91m')
        print('It appears that this is your first build since the migration to Ninja.')
        print('Please do a one-time manual removal of your target/ directory and try again.')
        print('\033[0m')


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


@dataclass
class Target:
    category: str
    name: str
    directory: str
    filename: str

    @staticmethod
    def parse(line: str) -> 'Target':
        tokens = line.split()
        assert len(tokens) == 4, line
        return Target(tokens[0], tokens[1], tokens[2], tokens[3])


def get_targets(repo_root, target_dir, specified_targets) -> List[Target]:
    targets_file = os.path.join(repo_root, target_dir, 'targets.txt')
    if not os.path.exists(targets_file):
        raise Exception(f'Targets file not found: {targets_file}')

    with open(targets_file, 'r') as f:
        all_targets = [Target.parse(line) for line in f.readlines()]

    if specified_targets:
        out1 = [t for t in all_targets if t.name in specified_targets]
        out2 = [t for t in all_targets if t.category in specified_targets]
        return out1 + out2
    else:
        return all_targets


def get_torch_dir():
    torch_dir = '/workspace/libtorch/current'
    assert os.path.isdir(torch_dir)
    return torch_dir


def main():
    cwd = os.getcwd()
    args = get_args()

    if args.clear_core_dumps:
        run('rm -f core.*')

    validate_gcc_version()
    validate_gxx_version()

    debug = bool(args.debug)

    targets = args.target.split(',') if args.target else []
    repo_root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(repo_root)

    # build extra deps unconditionally
    run('cd extra_deps && ./build.py')

    torch_dir = get_torch_dir()
    eigenrand_dir = os.path.join(repo_root, 'extra_deps/EigenRand')

    macro_defines = args.macro_defines if args.macro_defines else []
    macro_defines = [f'{d}=1' if d.find('=') == -1 else d for d in macro_defines]
    if debug:
        macro_defines.append('DEBUG_BUILD=1')

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
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f'-DMY_TORCH_DIR={torch_dir}',
        f'-DMY_EIGENRAND_DIR={eigenrand_dir}',
        f'-DEXTRA_DEFINITIONS="{extra_definitions}"',
    ]
    if debug:
        cmake_cmd_tokens.append('-DCMAKE_BUILD_TYPE=Debug')
    else:
        cmake_cmd_tokens.append('-DCMAKE_BUILD_TYPE=Release')

    cmake_cmd = ' '.join(cmake_cmd_tokens)
    run(cmake_cmd, handler=catch_first_time_ninja_error)

    # Only use half of the available CPU cores for building to avoid overloading the machine
    num_parallel_jobs = max(1, os.cpu_count() // 2)

    expanded_targets = get_targets(repo_root, target_dir, targets)
    os.chdir(target_dir)
    build_cmd_tokens = [
        'cmake',
        '--build',
        '.',
        '--parallel', str(num_parallel_jobs),
    ]
    if not targets:
        build_cmd_tokens.extend(['--target', 'all'])
    else:
        for t in expanded_targets:
            build_cmd_tokens.extend(['--target', t.name])

    build_cmd = ' '.join(build_cmd_tokens)
    run(build_cmd)

    bin_dir = os.path.join(repo_root, target_dir, 'bin')

    categories = set(t.category for t in expanded_targets)
    for category in categories:
        spec = GAME_SPECS_BY_NAME.get(category, None)
        if spec is None:
            continue
        extra_deps = spec.extra_runtime_deps
        for dep in extra_deps:
            dep_loc = os.path.join(repo_root, dep)
            if not os.path.exists(dep_loc):
                print(f'ERROR: extra dependency for {category} not found: {dep}')
                print('Please rerun setup_wizard.py to fix this.')
                raise Exception()
            extra_dir = os.path.join(bin_dir, 'extra')
            os.makedirs(extra_dir, exist_ok=True)
            cp_loc = extra_dir
            run(f'rsync -r {dep_loc} {cp_loc}', print_cmd=False)
            print(f'Extra dependency:', os.path.join(cp_loc, os.path.split(dep)[1]))

    expanded_targets.sort(key=lambda t: (t.directory, t.filename))
    for tgt in expanded_targets:
        full_path = os.path.join(tgt.directory, tgt.filename)
        if not os.path.isfile(full_path):
            continue
        path = os.path.relpath(full_path, cwd)
        print(path)


if __name__ == '__main__':
    main()
