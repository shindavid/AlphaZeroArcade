from .build_params import BuildParams

from alphazero.logic.custom_types import Generation
from games.game_spec import GameSpec
from shared.net_modules import ShapeInfo, ShapeInfoDict
from util.logging_util import get_logger
from util.repo_util import Repo

import torch

from cffi import FFI
from dataclasses import dataclass
import os
from typing import Iterable, List, Optional


logger = get_logger()


@dataclass
class DataBatch:
    input_tensor: torch.Tensor
    target_tensors: List[torch.Tensor]
    target_masks: List[torch.Tensor]


class GameLogReader:
    """
    Provides a wrapper around the game-specific shared library used to read game logs.

    The functions of the shared library are exposed through the FFI interface.
    """
    def __init__(self, game_spec: GameSpec, build_params: BuildParams,
                 cuda_device_str: str):
        self._game_spec = game_spec
        self._build_params = build_params
        self._cuda_device_str = cuda_device_str
        self._ffi = self._get_ffi()
        self._lib = self._get_shared_lib()
        self._lib.init()
        self._shape_info_dict: Optional[ShapeInfoDict] = None
        self._data_loader = None

    def close(self):
        if self._data_loader is not None:
            self._lib.DataLoader_delete(self._data_loader)
        self._ffi.dlclose(self._lib)

    def init_data_loader(self, data_dir: str,
                         memory_budget: int = 2**28,  # 256 MB
                         num_worker_threads=4,
                         num_prefetch_threads=4):
        ffi = self._ffi
        lib = self._lib

        data_dir = ffi.new('char[]', data_dir.encode('utf-8'))
        self._data_loader = lib.DataLoader_new(data_dir, memory_budget, num_worker_threads,
                                               num_prefetch_threads)

    @property
    def shape_info_dict(self) -> ShapeInfoDict:
        if self._shape_info_dict is None:
            self._shape_info_dict = self._load_shape_info_dict()
        return self._shape_info_dict

    def restore_data_loader(self, gens: List[Generation], row_counts: List[int],
                            file_sizes: List[int]):
        ffi = self._ffi
        lib = self._lib

        assert len(gens) == len(row_counts) == len(file_sizes)

        n = len(gens)
        if n == 0:
            return

        gens_c = ffi.new('int[]', gens)
        row_counts_c = ffi.new('int[]', row_counts)
        file_sizes_c = ffi.new('int64_t[]', file_sizes)
        lib.DataLoader_restore(self._data_loader, n, gens_c, row_counts_c, file_sizes_c)

    def add_gen(self, gen: Generation, num_rows: int, file_size: int):
        self._lib.DataLoader_add_gen(self._data_loader, gen, num_rows, file_size)

    def create_data_batches(self, window_size: int, minibatch_size: int, n_minibatches: int,
                            master_size: int, target_names: List[str], gen: Generation,
                            apply_symmetry=True) -> Iterable[DataBatch]:
        ffi = self._ffi
        lib = self._lib

        n_samples = minibatch_size * n_minibatches

        input_shape_info = self.shape_info_dict['input']
        target_shape_infos = [self.shape_info_dict[name] for name in target_names]

        input_shape = tuple([n_samples] + list(input_shape_info.shape))
        input_tensor = torch.empty(input_shape, dtype=torch.float32)

        target_shapes = [tuple([n_samples] + list(shape_info.shape)) for shape_info in
                         target_shape_infos]
        target_tensors = [torch.empty(shape, dtype=torch.float32) for shape in target_shapes]

        target_masks = [torch.empty(n_samples, dtype=torch.bool) for _ in target_shape_infos]

        target_indices = [s.target_index for s in target_shape_infos] + [-1]  # -1: null-terminator
        target_values = [ffi.cast('float*', tensor.data_ptr()) for tensor in target_tensors]
        target_mask_values = [ffi.cast('bool*', tensor.data_ptr()) for tensor in target_masks]

        input_values_c = ffi.cast('float*', input_tensor.data_ptr())
        target_indices_c = ffi.new('int[]', target_indices)
        target_values_c = ffi.new('float*[]', target_values)
        target_masks_c = ffi.new('bool*[]', target_mask_values)

        start_gen_tensor = torch.empty(1, dtype=torch.int32)
        start_gen_value_c = ffi.cast('int*', start_gen_tensor.data_ptr())

        logger.info('******************************')
        logger.info('Train gen:%s', gen)

        lib.DataLoader_load(self._data_loader, window_size, n_samples, apply_symmetry,
                            input_values_c, target_indices_c, target_values_c, target_masks_c,
                            start_gen_value_c)

        start_gen = start_gen_tensor.item()

        if start_gen + 1 == gen:
            gen_str = f'gen {start_gen}'
        else:
            gen_str = f'gens {start_gen} to {gen - 1}'
        logger.info(
            'Sampling from %s of %s (%.1f%%) positions (%s)' %
            (window_size, master_size, 100. * window_size / master_size, gen_str))

        input_tensor = input_tensor.to(self._cuda_device_str)
        for t, target_tensor in enumerate(target_tensors):
            target_tensors[t] = target_tensor.to(self._cuda_device_str)
            target_masks[t] = target_masks[t].to(self._cuda_device_str)

        for i in range(n_minibatches):
            start = i * minibatch_size
            end = (i + 1) * minibatch_size

            input_tensor_slice = input_tensor[start:end]
            target_tensor_slices = [tensor[start:end] for tensor in target_tensors]
            target_mask_slices = [mask[start:end] for mask in target_masks]

            data_batch = DataBatch(input_tensor=input_tensor_slice,
                                   target_tensors=target_tensor_slices,
                                   target_masks=target_mask_slices)
            yield data_batch

    def _get_ffi(self):
        ffi = FFI()
        ffi.cdef("""
            struct DataLoader {};

            struct ShapeInfo {
                char* name;
                int* dims;
                int num_dims;
                int target_index;
            };

            struct ShapeInfo* get_shape_info_array();

            void free_shape_info_array(struct ShapeInfo* info);

            struct DataLoader* DataLoader_new(const char* data_dir, int64_t memory_budget,
                int num_worker_threads, int num_prefetch_threads);

            void DataLoader_delete(struct DataLoader* loader);

            void DataLoader_restore(struct DataLoader* loader, int n, int* gens, int* row_counts,
                int64_t* file_sizes);

            void DataLoader_add_gen(struct DataLoader* loader, int gen, int num_rows,
                int64_t file_size);

            void DataLoader_load(struct DataLoader* loader, int64_t window_size, int n_samples,
                bool apply_symmetry, float* input_data_array, int* target_indices_array,
                float** target_data_arrays, bool** target_mask_arrays, int* start_gen);

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
