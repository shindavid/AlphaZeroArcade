#!/usr/bin/env python3

import os
import sys

from config import Config


def run(cmd):
    print(cmd)
    if os.system(cmd):
        sys.exit(1)


torch_dir = Config.instance().get('libtorch_dir')

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
cfg = Config.instance().filename
torch_link = 'https://pytorch.org/get-started/locally/'
assert torch_dir, (f'Please add an entry for "libtorch_dir" to {cfg}. Can download it from {torch_link}. '
                   'An explicitly downloaded libtorch must be used instead of the one that ships with conda, '
                   'due to the cxx11-abi issue. See here: https://github.com/pytorch/pytorch/issues/17492')

conda_prefix = os.environ.get('CONDA_PREFIX', None)
assert conda_prefix, 'It appears you do not have a conda environment activated. Please activate!'
eigen_dir = os.path.join(conda_prefix, 'share/eigen3/cmake')
assert os.path.isdir(eigen_dir), 'Please conda install eigen.'

os.chdir(repo_root)
cmake_cmd = f'cmake CMakeLists.txt -Btarget -DMY_TORCH_DIR={torch_dir} -DCMAKE_PREFIX_PATH={conda_prefix}'
run(cmake_cmd)

os.chdir('target')
run('make')
