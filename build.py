#!/usr/bin/env python3

import os
import sys
import torch


def run(cmd):
    print(cmd)
    if os.system(cmd):
        sys.exit(1)


os.chdir(os.path.dirname(__file__))

torch_cmake_path = torch.utils.cmake_prefix_path
torch_dir = os.path.dirname(os.path.dirname(torch_cmake_path))

cmake_cmd = f'cmake CMakeLists.txt -Btarget -DCMAKE_PREFIX_PATH="{torch_dir}" -DEigen3_DIR=/home/dshin/anaconda3/envs/dev/share/eigen3/cmake'
run(cmake_cmd)

os.chdir('target')
run('make')

