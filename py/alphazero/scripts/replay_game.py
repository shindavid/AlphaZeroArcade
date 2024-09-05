#!/usr/bin/env python3

import argparse
from cffi import FFI

import torch

from alphazero.logic.build_params import BuildParams
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer


torch.set_printoptions(linewidth=120, sci_mode=False)


def load_args():
    parser = argparse.ArgumentParser()

    RunParams.add_args(parser)
    BuildParams.add_args(parser, add_binary_path_option=False)

    group = parser.add_argument_group('replay_game.py options')

    group.add_argument('-G', '--gen', type=int, help='generation (default: latest)')
    group.add_argument('-s', '--self-play-filename', help='self-play filename. If not specified, '
                       'picks one from $A0A_OUTPUT_DIR/{game}/{tag}/self-play-data/*/gen-{gen}/')

    return parser.parse_args()


def main():
    args = load_args()
    run_params = RunParams.create(args, require_tag=False)
    build_params = BuildParams.create(args)
    gen = args.gen
    self_play_filename = args.self_play_filename

    if self_play_filename is not None:
        assert gen is None, 'Cannot specify both --gen/-G and --self-play-filename/-s'
        assert not run_params.tag, 'Cannot specify both --tag/-t and --self-play-filename/-s'
    else:
        assert run_params.tag, 'Must specify either --tag/-t or --self-play-filename/-s'

        organizer = DirectoryOrganizer(run_params)
        self_play_filename = organizer.get_any_self_play_data_filename(gen)

    print(f'Replaying from {self_play_filename}\n')

    ffi = FFI()

    lib = ffi.dlopen(build_params.get_ffi_lib_path(run_params.game))

    # Declare the functions and types used
    ffi.cdef("""
        struct GameLog {};
        struct GameLog* GameLog_new(const char* filename);
        void GameLog_delete(struct GameLog* log);
        void GameLog_replay(struct GameLog* log);
    """)

    filename_c = ffi.new('char[]', self_play_filename.encode('utf-8'))

    log = lib.GameLog_new(filename_c)
    lib.GameLog_replay(log)
    lib.GameLog_delete(log)

    print(f'Finished replaying from {self_play_filename}')


if __name__ == '__main__':
    main()
