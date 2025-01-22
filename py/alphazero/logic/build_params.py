import argparse
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BuildParams:
    """
    Parameters used to specify built files for a particular AlphaZero run.
    """
    debug_build: Optional[bool] = None
    binary_path: Optional[str] = None
    ffi_lib_path: Optional[str] = None

    def get_binary_path(self, game: str):
        if self.binary_path:
            return self.binary_path

        build = 'Debug' if self.debug_build else 'Release'
        return f'target/{build}/bin/{game}'

    def get_ffi_lib_path(self, game: str):
        if self.ffi_lib_path:
            return self.ffi_lib_path

        # TODO[dshin]: The ffi-debug build does not work. This is due to some hairy issues with
        # libtorch and dynamic library loading. For now, we will just use the Release build.
        #
        # My hope is to eventually retire the libtorch-dependency in favor of something like
        # onnxruntime, in which case this problem should disappear. So I'm punting on a proper fix
        # for now.
        build = 'Release'
        # build = 'Debug' if self.debug_build else 'Release'
        return f'target/{build}/lib/lib{game}.so'

    @staticmethod
    def create(args) -> 'BuildParams':
        params = BuildParams(
            debug_build=args.debug_build,
            binary_path=getattr(args, 'binary_path', None),
            ffi_lib_path=getattr(args, 'ffi_lib_path', None),
        )
        return params

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, add_binary_path_option=True,
                 add_ffi_lib_path_option=True):
        group = parser.add_argument_group('Build options')

        group.add_argument('-d', '--debug-build', action='store_true',
                           help='use Debug binary (default: Release)')
        if add_binary_path_option:
            group.add_argument(
                '--binary-path', help='path to binary (default: target/{Debug,Release}/bin/{game})')
        if add_ffi_lib_path_option:
            group.add_argument(
                '--ffi-lib-path',
                help='path to ffi library (default: target/Release/lib/lib{game}.so)')

    def add_to_cmd(self, cmd: List[str], add_binary_path_option=True,
                   add_ffi_lib_path_option=True):
        if self.debug_build:
            cmd.append('--debug-build')
        if add_binary_path_option and self.binary_path:
            cmd.extend(['--binary-path', self.binary_path])
        if add_ffi_lib_path_option and self.ffi_lib_path:
            cmd.extend(['--ffi-lib-path', self.ffi_lib_path])
