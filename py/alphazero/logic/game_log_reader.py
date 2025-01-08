from .build_params import BuildParams

from games.game_spec import GameSpec
from shared.net_modules import ShapeInfo, ShapeInfoDict
from util.logging_util import get_logger
from util.repo_util import Repo

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

    def __init__(self, game_spec: GameSpec, build_params: BuildParams):
        self._game_spec = game_spec
        self._build_params = build_params
        self._ffi = self._get_ffi()
        self._lib = self._get_shared_lib()
        self._lib.init()
        self._shape_info_dict: Optional[ShapeInfoDict] = None

    def close(self):
        self._ffi.dlclose(self._lib)

    @property
    def shape_info_dict(self) -> ShapeInfoDict:
        if self._shape_info_dict is None:
            self._shape_info_dict = self._load_shape_info_dict()
        return self._shape_info_dict

    def open_log(self, filename: str):
        ffi = self._ffi
        filename = ffi.new('char[]', filename.encode('utf-8'))
        return self._lib.GameLog_new(filename)

    def num_sampled_positions(self, log):
        return self._lib.GameLog_num_sampled_positions(log)

    def close_log(self, log):
        self._lib.GameLog_delete(log)

    def create_tensors(self, log, input_shape_info: ShapeInfo, target_shape_infos: List[ShapeInfo],
                       index: int, apply_symmetry: bool = True):
        ffi = self._ffi
        lib = self._lib

        input_tensor = torch.empty(input_shape_info.shape, dtype=torch.float32)
        target_tensors = [torch.empty(shape_info.shape, dtype=torch.float32)
                          for shape_info in target_shape_infos]
        target_masks = [torch.empty(1, dtype=torch.bool) for _ in target_shape_infos]
        target_indices = [s.target_index for s in target_shape_infos] + [-1]  # -1: null-terminator
        target_values = [ffi.cast('float*', tensor.data_ptr()) for tensor in target_tensors]
        target_mask_values = [ffi.cast('bool*', tensor.data_ptr()) for tensor in target_masks]

        input_values_c = ffi.cast('float*', input_tensor.data_ptr())
        target_indices_c = ffi.new('int[]', target_indices)
        target_values_c = ffi.new('float*[]', target_values)
        target_masks_c = ffi.new('bool*[]', target_mask_values)

        lib.GameLog_load(log, index, apply_symmetry, input_values_c, target_indices_c,
                         target_values_c, target_masks_c)
        return [input_tensor] + target_tensors + target_masks

    def _get_ffi(self):
        ffi = FFI()
        ffi.cdef("""
            struct GameLog {};

            struct ShapeInfo {
                char* name;
                int* dims;
                int num_dims;
                int target_index;
            };

            struct ShapeInfo* get_shape_info_array();
            void free_shape_info_array(struct ShapeInfo* info);

            struct GameLog* GameLog_new(const char* filename);
            void GameLog_delete(struct GameLog* log);
            void GameLog_load(struct GameLog* log, int index, bool apply_symmetry,
                       float* input_values, int* target_indices, float** target_value_arrays,
                       bool** target_masks);
            int GameLog_num_sampled_positions(struct GameLog* log);

            void init();
            """)
        return ffi

    def _get_shared_lib(self) -> str:
        name = self._game_spec.name

        shared_lib = os.path.join(Repo.root(), self._build_params.get_ffi_lib_path(name))
        assert os.path.isfile(shared_lib), f'Could not find shared lib: {shared_lib}'
        return self._ffi.dlopen(shared_lib)

    def _load_shape_info_dict(self) -> ShapeInfoDict:
        ffi = self._ffi
        lib = self._lib

        shape_info_arr = lib.get_shape_info_array()

        shape_info_dict = {}
        i = 0
        while True:
            info = shape_info_arr[i]
            if info.name == ffi.NULL:
                break
            name = ffi.string(info.name).decode('utf-8')
            shape = tuple([info.dims[j] for j in range(info.num_dims)])
            shape_info = ShapeInfo(name=name, target_index=info.target_index, shape=shape)
            shape_info_dict[name] = shape_info
            logger.debug('ShapeInfo: %s -> %s', name, shape_info)
            i += 1

        lib.free_shape_info_array(shape_info_arr)
        return shape_info_dict
