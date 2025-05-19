import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BuildParams:
    """
    Parameters used to specify built files for a particular AlphaZero run.
    """
    debug_build: Optional[bool] = None
    binary_path: Optional[str] = None
    ffi_lib_path: Optional[str] = None
    override_binary: bool = False
    use_stored_binary: bool = False

    def __post_init__(self):
        if self.binary_path is not None:
            if self.debug_build is not None:
                raise ValueError(
                     f'--binary-path {self.binary_path} is provided; using --debug-build is not allowed.')

            if self.override_binary:
                raise ValueError(
                    f'--binary-path {self.binary_path} is provided; cannot override binaries.')

        if self.use_stored_binary:
            if self.binary_path is not None:
                raise ValueError(
                    f'--binary-path {self.binary_path} is provided; cannot use stored binary.')

            if self.debug_build:
                raise ValueError('--debug-build and --use-stored-binary are mutually exclusive.')

    @property
    def build_type(self):
        return 'Debug' if self.debug_build else 'Release'

    def get_binary_path(self, game: str):
        if self.binary_path:
            return self.binary_path
        return f'target/{self.build_type}/bin/{game}'

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
            override_binary=getattr(args, 'override_binary', defaults.override_binary),
            use_stored_binary=getattr(args, 'use_stored_binary', defaults.use_stored_binary),
        )
        return params

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, loop_controller=False):
        group = parser.add_argument_group('Build options')

        if loop_controller:
            group.add_argument(
                '-d', '--debug-build', action='store_true',
                help='use Debug binary. Ignored if --binary-path is specified')
            group.add_argument(
                '--binary-path',
                help='path to binary (default: target/{Debug,Release}/bin/{game})')
            group.add_argument(
                '--ffi-lib-path',
                help='path to ffi library (default: target/Release/lib/lib{game}.so)')
            group.add_argument(
                '--override-binary', action='store_true',
                help='override the binary file in output/{game}/{tag}/target/{Debug, Release}')
            group.add_argument(
                '--use-stored-binary', action='store_true',
                help='Use the stored binary instead of the one in target/Release')
        else:
            group.add_argument(
                '--binary-path',
                help='path to binary (default: use binary received from loop controller)')

    def add_to_cmd(self, cmd: List[str], loop_controller=False):
        defaults = BuildParams()

        if self.binary_path != defaults.binary_path:
            cmd.extend(['--binary-path', self.binary_path])

        if loop_controller:
            if self.debug_build:
                cmd.append('--debug-build')
            if self.ffi_lib_path != defaults.ffi_lib_path:
                cmd.extend(['--ffi-lib-path', self.ffi_lib_path])
            if self.override_binary:
                cmd.append('--override-binary')
            if self.use_stored_binary:
                cmd.append('--use-stored-binary')
