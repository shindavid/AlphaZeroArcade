from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import create_symlink, copy_file_to_folder, CustomHelpFormatter

import argparse
import os


"""
This script copies the binaries from the release directory to the run directory. It creates
symlinks for the extra dependencies in the release directory. This needs to be run manually
before benchmarker, evaulator or mctsevaluator.

Note:
- This functionality should be integrated into the existing scripts instead of requiring a manual step.
- Only the loop-controller has access to the directory, so it should be responsible for copying the binaries.
- However, currently, the self-play server and ratings server specify their own binaries at launch.
- The proper solution would be for the loop-controller to send the binary over TCP to these servers.

TODO:
- Implement this in the future when restructuring the process.
- We should replace the creation of symlinks with copying/sending the files.
"""
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

