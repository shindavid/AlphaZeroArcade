#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import os
import subprocess
import sys
from typing import List

from games.index import GAME_SPECS_BY_NAME
from util.py_util import CustomHelpFormatter


def run(cmd: str, print_cmd=True):
    if print_cmd:
        print(cmd)

    proc = subprocess.Popen(cmd, shell=True, text=True, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    if proc.returncode:
        print(stderr)
        sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument("-d", '--debug', action='store_true', help='debug build')
    parser.add_argument('--enable-debug-logging', action='store_true', help='debug logging')
    parser.add_argument('--clean', action='store_true', help='clean out target/.../{bin,lib}/ directory')
    parser.add_argument("-c", '--clear-core-dumps', action='store_true', help='rm core.* (in cwd) before doing anything')
    parser.add_argument("-t", '--target', help='build targets, comma-separated. Default: all')
    parser.add_argument('-D', '--macro-defines', action='append',
                        help='macro definitions to forward to make cmd (-D FOO -D BAR=2). If a macro name is passed'
                        ' without an assigned value, it is given a value of "1" by default. This plays nicely with the'
                        ' IS_MACRO_ENABLED() macro function defined in cpp/util/CppUtil.hpp')
    return parser.parse_args()


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


def main():
    cwd = os.getcwd()
    args = get_args()

    if args.clear_core_dumps:
        run('rm -f core.*')

    debug = bool(args.debug)
    enable_debug_logging = bool(args.enable_debug_logging)

    targets = args.target.split(',') if args.target else []
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    # build extra deps unconditionally
    run('cd extra_deps && ./build.py')

    eigenrand_dir = os.path.join(repo_root, 'extra_deps/EigenRand')

    macro_defines = args.macro_defines if args.macro_defines else []
    macro_defines = [f'{d}=1' if d.find('=') == -1 else d for d in macro_defines]
    if debug:
        macro_defines.append('DEBUG_BUILD=1')
    if enable_debug_logging:
        macro_defines.append('SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG')

    extra_definitions = ' '.join(f'-D{d}' for d in macro_defines)

    build_name = 'Debug' if debug else 'Release'
    target_dir = f'target/{build_name}'

    if args.clean:
        run(f'rm -rf {target_dir}/bin/*')
        run(f'rm -rf {target_dir}/lib/*')

    cmake_cmd_tokens = [
        'cmake',
        '-G Ninja',
        'CMakeLists.txt',
        f'-B{target_dir}',
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f'-DMY_EIGENRAND_DIR={eigenrand_dir}',
        f'-DEXTRA_DEFINITIONS="{extra_definitions}"',
    ]
    if debug:
        cmake_cmd_tokens.append('-DCMAKE_BUILD_TYPE=Debug')
    else:
        cmake_cmd_tokens.append('-DCMAKE_BUILD_TYPE=Release')

    cmake_cmd = ' '.join(cmake_cmd_tokens)
    run(cmake_cmd)

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

    expanded_targets.sort(key=lambda t: (t.directory, t.filename))
    for tgt in expanded_targets:
        full_path = os.path.join(tgt.directory, tgt.filename)
        if not os.path.isfile(full_path):
            continue
        path = os.path.relpath(full_path, cwd)
        print(path)


if __name__ == '__main__':
    main()
