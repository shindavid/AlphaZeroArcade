#!/usr/bin/env python3

# Usage: ./binary_dump_game_log.py <game_log_path>
#
# This is equivalent to running:
#
# xxd -e <game_log_path> | cut -d' ' -f1-6

import sys
import subprocess

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <game_log_path>')
    sys.exit(1)

cmd = f'xxd -e {sys.argv[1]} | cut -d" " -f1-6'
subprocess.run(cmd, shell=True)
