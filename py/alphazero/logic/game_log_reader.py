from games.game_spec import GameSpec
from util.logging_util import get_logger
from util.repo_util import Repo
from util.torch_util import ShapeDict

import torch

from cffi import FFI
import os
from typing import List, Optional


logger = get_logger()


class GameLogReader:
    """
    Provides a wrapper around the game-specific shared library used to read game logs.

    The functions of the shared library are exposed through the FFI interface.
    """

    def __init__(self, game_spec: GameSpec):
        self._game_spec = game_spec
        self._ffi = self._get_ffi()
        self._lib = self._get_shared_lib()
        self._shape_info: Optional[ShapeDict] = None

    def close(self):
        self._ffi.dlclose(self._lib)

    @property
    def shape_info(self) -> ShapeDict:
        if self._shape_info is None:
            self._shape_info = self._load_shape_info()
        return self._shape_info

    def open_log(self, filename: str):
        ffi = self._ffi
        filename = ffi.new('char[]', filename.encode('utf-8'))
        return self._lib.GameLogReader_new(filename)

    def close_log(self, log):
        self._lib.GameLogReader_delete(log)

    def create_tensors(self, log, names: List[str], index: int, apply_symmetry: bool = True):
        ffi = self._ffi
        lib = self._lib

        tensors = [torch.empty(self.shape_info[key], dtype=torch.float32) for key in names]

        keys = [ffi.new("char[]", key.encode('utf-8')) for key in names]
        values = [ffi.cast("float *", arr.data_ptr()) for arr in tensors]

        keys_c = ffi.new("const char *[]", keys)
        values_c = ffi.new("float *[]", values)

        lib.GameLogReader_load(log, index, apply_symmetry, keys_c, values_c, len(keys))
        return tensors

    def _get_ffi(self):
        ffi = FFI()
        ffi.cdef("""
            struct GameLogReader {};

            struct ShapeInfo {
                char* name;
                int* dims;
                int num_dims;
            };

            struct ShapeInfo* get_shape_info_array();
            void free_shape_info_array(struct ShapeInfo* info);

            struct GameLogReader* GameLogReader_new(const char* filename);
            void GameLogReader_delete(struct GameLogReader* log);
            void GameLogReader_load(struct GameLogReader* log, int index, bool apply_symmetry,
                       const char** keys, float** values, int num_keys);
            """)
        return ffi

    def _get_shared_lib(self) -> str:
        name = self._game_spec.name
        shared_lib = os.path.join(Repo.root(), 'target/Release/lib', f'lib{name}.so')
        assert os.path.isfile(shared_lib), f'Could not find shared lib: {shared_lib}'
        return self._ffi.dlopen(shared_lib)

    def _load_shape_info(self) -> ShapeDict:
        ffi = self._ffi
        lib = self._lib

        shape_info_arr = lib.get_shape_info_array()

        shape_info = {}
        i = 0
        while True:
            info = shape_info_arr[i]
            if info.name == ffi.NULL:
                break
            name = ffi.string(info.name).decode('utf-8')
            dims = tuple([info.dims[j] for j in range(info.num_dims)])
            shape_info[name] = dims
            logger.debug(f'Tensor shape: {name} -> {dims}')
            i += 1

        lib.free_shape_info_array(shape_info_arr)
        return shape_info
