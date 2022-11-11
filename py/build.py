#!/usr/bin/env python3
import argparse
import os
import sys

from config import Config


def run(cmd):
    print(cmd)
    if os.system(cmd):
        sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--debug', action='store_true', help='debug build')
    return parser.parse_args()


def main():
    args = get_args()
    debug = bool(args.debug)

    torch_dir = Config.instance().get('libtorch_dir')

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg = Config.instance().filename
    torch_link = 'https://pytorch.org/get-started/locally/'
    assert torch_dir, (f'Please download torch for C++ from {torch_link}, unzip to a path LIBTORCH_PATH, and then add an '
                       f'entry "libtorch_dir=LIBTORCH_PATH" to {cfg}. '
                       'An explicitly downloaded libtorch must be used instead of the one that ships with conda, '
                       'due to the cxx11-abi issue. See here: https://github.com/pytorch/pytorch/issues/17492')
    assert os.path.isdir(torch_dir)

    conda_prefix = os.environ.get('CONDA_PREFIX', None)
    assert conda_prefix, 'It appears you do not have a conda environment activated. Please activate!'
    eigen_dir = os.path.join(conda_prefix, 'share/eigen3/cmake')
    assert os.path.isdir(eigen_dir), 'Please conda install eigen.'

    lib_cmake_dir = os.path.join(conda_prefix, 'lib/cmake')
    assert os.path.isdir(lib_cmake_dir)
    found_boost = False
    for path in os.listdir(lib_cmake_dir):
        if path.startswith('Boost-') and os.path.isdir(os.path.join(lib_cmake_dir, path)):
            found_boost = True
            break
    assert found_boost, 'Please conda install boost.'

    os.chdir(repo_root)

    build_name = 'Debug' if debug else 'Release'
    target_dir = f'target/{build_name}'
    cmake_cmd_tokens = [
        'cmake',
        'CMakeLists.txt',
        f'-B{target_dir}',
        f'-DMY_TORCH_DIR={torch_dir}',
        f'-DCMAKE_PREFIX_PATH={conda_prefix}'
    ]
    if debug:
        cmake_cmd_tokens.append('-DCMAKE_BUILD_TYPE=Debug')
    cmake_cmd = ' '.join(cmake_cmd_tokens)
    run(cmake_cmd)

    os.chdir(target_dir)
    run('make -j2')  # currently we have 2 main targets
    bin_postfix = 'd' if debug else ''
    bin_loc1 = os.path.join(repo_root, target_dir, 'bin', f'c4_gen_training_data{bin_postfix}')
    bin_loc2 = os.path.join(repo_root, target_dir, 'bin', f'c4_play_vs_cpu{bin_postfix}')
    print(f'Binary location: {bin_loc1}')
    print(f'Binary location: {bin_loc2}')


if __name__ == '__main__':
    main()
