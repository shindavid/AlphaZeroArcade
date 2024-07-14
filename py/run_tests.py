#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

from util.repo_util import Repo
from util import subprocess_util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--build', help='Release or Debug. Default: most recent build')
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


def run_tests(build):
    tests_dir = f'target/{build}/bin/tests'

    if not os.path.isdir(tests_dir):
        print(f'No built tests found for {build}. Please run py/build.py first.')
        sys.exit(1)

    bins = os.listdir(tests_dir)

    if not bins:
        print(f'No built tests found for {build}. Please run py/build.py first.')
        sys.exit(1)

    pass_count = 0
    fail_count = 0
    for bin in bins:
        full_bin = os.path.join(tests_dir, bin)
        print(f'Running: {full_bin}')

        proc = subprocess_util.Popen(full_bin)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            print('FAILURE!')
            print('stdout:')
            print(stdout)
            print('stderr:')
            print(stderr)
            fail_count += 1
        else:
            pass_count += 1

    if fail_count == 0:
        print('All tests passed!')
        sys.exit(0)
    else:
        print(f'Failed {fail_count} of {fail_count + pass_count} tests!')
        sys.exit(1)


def main():
    args = get_args()

    os.chdir(Repo.root())
    build = args.build
    assert build in ('Release', 'Debug', None), f'Invalid --build/-b argument: {build}'

    if build is None:
        build = get_default_build()

    run_tests(build)


if __name__ == '__main__':
    main()
