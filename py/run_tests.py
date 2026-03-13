#!/usr/bin/env python3
from util.repo_util import Repo
from util import subprocess_util
from util.py_util import CustomHelpFormatter

from termcolor import colored

import argparse
import json
import os
import sys

import torch


def get_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument('-b', '--build', help='Release or Debug. Default: most recent build')
    parser.add_argument('-c', '--cpp-only', action='store_true',
                        help='Run only C++ tests, skip Python tests')
    parser.add_argument('-p', '--py-only', action='store_true',
                        help='Run only Python tests, skip C++ tests')
    parser.add_argument('-w', '--write-goldenfiles', action='store_true',
                        help='Write goldenfiles (passed as --write-goldenfiles to C++ test binaries)')
    return parser.parse_args()


def get_build_time(tests_dir):
    if not os.path.isdir(tests_dir):
        return 0

    mtimes = []
    for filename in os.listdir(tests_dir):
        full_filename = os.path.join(tests_dir, filename)
        if os.path.isfile(full_filename):
            mtimes.append(os.path.getmtime(full_filename))

    return max(mtimes) if mtimes else 0


def get_default_build():
    release_build_time = get_build_time('target/Release/bin/tests')
    debug_build_time = get_build_time('target/Debug/bin/tests')

    if release_build_time == 0 and debug_build_time == 0:
        print('No built tests found. Please run py/build.py first.')
        sys.exit(1)

    build = 'Release' if release_build_time > debug_build_time else 'Debug'
    print(f'Auto-detected build: {build}')
    return build


def get_cpp_bins(targets_file, tests_dir):
    if not os.path.exists(targets_file):
        raise Exception(f'Targets file not found: {targets_file}')

    with open(targets_file, 'r') as f:
        targets = json.load(f)

    bins = []
    for entry in targets:
        directory = entry['directory']
        if os.path.isdir(directory) and os.path.samefile(directory, tests_dir):
            bins.append(entry['filename'])
    return bins


def run_cpp_tests(build, write_goldenfiles=False):
    n = torch.cuda.device_count()
    assert n > 0, 'No GPU found. Try exiting and relaunching run_docker.py'

    build_dir = f'target/{build}'
    tests_dir = f'{build_dir}/bin/tests'

    if not os.path.isdir(tests_dir):
        print(colored(f'No built tests found for {build}. Please run py/build.py first.', 'red'))
        sys.exit(1)

    targets_file = os.path.join(build_dir, 'targets.json')
    bins = get_cpp_bins(targets_file, tests_dir)

    if not bins:
        print(colored(f'No built tests found for {build}. Please run py/build.py first.', 'red'))
        sys.exit(1)

    pass_count = 0
    fail_count = 0
    failed_tests = []

    for bin in bins:
        full_bin = os.path.join(tests_dir, bin)
        print(f'Running: {full_bin}')

        cmd = [full_bin]
        if write_goldenfiles:
            cmd.append('--write-goldenfiles')
        proc = subprocess_util.Popen(cmd)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            print(colored('FAILURE!', 'red'))
            print('stdout:')
            print(stdout)
            print('stderr:')
            print(stderr)
            fail_count += 1
            failed_tests.append(full_bin)
        else:
            pass_count += 1

    if fail_count == 0:
        print(colored(f'All {pass_count} c++ tests passed!', 'green'))
    else:
        print(colored(f'Failed {fail_count} of {fail_count + pass_count} c++ tests!', 'red'))
        for filename in failed_tests:
            print(colored(f"❌ {filename}", 'red'))


def run_py_tests():
    """
    Walks the py/unit_tests/ directory and runs every python file contained within.
    """
    tests_dir = 'py/unit_tests'

    pass_count = 0
    fail_count = 0
    failed_tests = []

    # use os.walk to find all python files in the tests directory
    for root, dirs, files in os.walk(tests_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # we found a python test file, run it
                full_file = os.path.join(root, file)
                print(f'Running: {full_file}')
                cmd = ['python3', full_file]
                proc = subprocess_util.Popen(cmd)
                stdout, stderr = proc.communicate()
                if proc.returncode:
                    print(colored(f'FAILURE in {full_file}!', 'red'))
                    print('stdout:')
                    print(stdout)
                    print('stderr:')
                    print(stderr)
                    fail_count += 1
                    failed_tests.append(full_file)
                else:
                    pass_count += 1

    if fail_count == 0:
        print(colored(f'All {pass_count} python tests passed!', 'green'))
    else:
        print(colored(f'Failed {fail_count} of {fail_count + pass_count} python tests!', 'red'))
        for filename in failed_tests:
            print(colored(f"❌ {filename}", 'red'))


def main():
    args = get_args()

    os.chdir(Repo.root())
    build = args.build
    assert build in ('Release', 'Debug', None), f'Invalid --build/-b argument: {build}'

    if build is None:
        build = get_default_build()

    if not args.py_only:
        run_cpp_tests(build, write_goldenfiles=args.write_goldenfiles)
    if not args.cpp_only:
        run_py_tests()


if __name__ == '__main__':
    main()
