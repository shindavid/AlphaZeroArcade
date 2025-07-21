#!/usr/bin/env python3

"""
This script runs clang-format on every file in the cpp/ directory, with the exception
of third-party code.
"""

import os
import subprocess
import sys


ROOT = os.path.join(os.path.dirname(__file__), '../cpp')
EXTENSIONS = {'.cpp', '.hpp', '.inl'}


def process(file_path):
    format_cmd = [
        'clang-format',
        '-i',
        file_path
    ]
    print(f"Running: {' '.join(format_cmd)}")
    subprocess.run(format_cmd, check=False, stdout=sys.stdout, stderr=sys.stderr)


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if 'third_party' in dirpath.split(os.sep):
            continue
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in EXTENSIONS:
                file_path = os.path.join(dirpath, filename)
                process(file_path)

if __name__ == '__main__':
    main()
