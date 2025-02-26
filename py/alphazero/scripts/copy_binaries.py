from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import create_symlink, copy_file_to_folder, CustomHelpFormatter

import argparse
import os


def copy_binaries(organizer: DirectoryOrganizer):
    binary_release_dir = '/workspace/repo/target/Release/bin/'
    copy_file_to_folder(os.path.join(binary_release_dir, organizer.game), organizer.base_dir)
    create_symlink(os.path.join(binary_release_dir, 'extra'), os.path.join(organizer.base_dir, 'extra'))


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    args = parser.parse_args()
    run_params = RunParams.create(args)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    copy_binaries(organizer)


if __name__ == '__main__':
    main()