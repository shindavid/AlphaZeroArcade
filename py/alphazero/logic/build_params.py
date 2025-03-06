import argparse
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class _BinaryPathArgument(Enum):
    default = 'default'


BinaryPathArgument = Union[str, None, _BinaryPathArgument]


@dataclass
class BuildParams:
    """
    Parameters used to specify built files for a particular AlphaZero run.
    """
    debug_build: Optional[bool] = None
    binary_path: Optional[str] = None
    ffi_lib_path: Optional[str] = None

    @property
    def build_type(self):
        return 'Debug' if self.debug_build else 'Release'

    def get_binary_path(self, game: str, default: BinaryPathArgument = _BinaryPathArgument.default):
        if self.binary_path:
            return self.binary_path

        if default == _BinaryPathArgument.default:
            return f'target/{self.build_type}/bin/{game}'

        return default

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
        # build = self.build_type
        return f'target/{build}/lib/lib{game}.so'

    @staticmethod
    def create(args) -> 'BuildParams':
        defaults = BuildParams()

        params = BuildParams(
            debug_build=getattr(args, 'debug_build', defaults.debug_build),
            binary_path=getattr(args, 'binary_path', defaults.binary_path),
            ffi_lib_path=getattr(args, 'ffi_lib_path', defaults.ffi_lib_path),
        )
        return params

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, add_binary_path_option=True,
                 add_ffi_lib_path_option=True, add_debug_build_option=True):
        group = parser.add_argument_group('Build options')

        if add_debug_build_option:
            group.add_argument(
                '-d', '--debug-build', action='store_true',
                help='use Debug binary. Ignored if --binary-path is specified')
        if add_binary_path_option:
            group.add_argument(
                '--binary-path',
                help='path to binary (default: target/{Debug,Release}/bin/{game})')
        if add_ffi_lib_path_option:
            group.add_argument(
                '--ffi-lib-path',
                help='path to ffi library (default: target/Release/lib/lib{game}.so)')

    def add_to_cmd(self, cmd: List[str], add_binary_path_option=True,
                   add_ffi_lib_path_option=True, add_debug_build_option=True):
        defaults = BuildParams()

        if add_debug_build_option and self.debug_build != defaults.debug_build:
            cmd.append('--debug-build')
        if add_binary_path_option and self.binary_path != defaults.binary_path:
            cmd.extend(['--binary-path', self.binary_path])
        if add_ffi_lib_path_option and self.ffi_lib_path != defaults.ffi_lib_path:
            cmd.extend(['--ffi-lib-path', self.ffi_lib_path])
