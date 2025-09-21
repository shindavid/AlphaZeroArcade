from .build_params import BuildParams

from alphazero.logic.custom_types import Generation
from games.game_spec import GameSpec
from shared.basic_types import SearchParadigm, ShapeInfo, ShapeInfoDict
from util.repo_util import Repo

import torch

from cffi import FFI
from dataclasses import dataclass
import logging
import math
import os
from typing import Iterable, List, Optional


logger = logging.getLogger(__name__)


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
                 cuda_device_str: str, paradigm: SearchParadigm):
        self._game_spec = game_spec
        self._build_params = build_params
        self._cuda_device_str = cuda_device_str
        self._paradigm = paradigm
        self._ffi = self._get_ffi()
        self._lib = self._get_shared_lib()
        self._lib.init()
        self._shape_info_dict: Optional[ShapeInfoDict] = None
        self._data_loader = None
        self._closed = False

    def closed(self) -> bool:
        return self._closed

    def close(self):
        self._closed = True
        if self._data_loader is not None:
            self._lib.DataLoader_delete(self._data_loader)
        self._ffi.dlclose(self._lib)

    def init_data_loader(self, data_dir: str,
                         memory_budget: int = 2**28,  # 256 MB
                         num_worker_threads=4,
                         num_prefetch_threads=4):
        ffi = self._ffi
        lib = self._lib

        data_dir_c = ffi.new('char[]', data_dir.encode('utf-8'))
        paradigm_str_c = ffi.new('char[]', self._paradigm.value.encode('utf-8'))

        self._data_loader = lib.DataLoader_new(data_dir_c, memory_budget, num_worker_threads,
                                               num_prefetch_threads, paradigm_str_c)

    @property
    def shape_info_dict(self) -> ShapeInfoDict:
        if self._shape_info_dict is None:
            self._shape_info_dict = self._load_shape_info_dict()
        return self._shape_info_dict

    def merge_game_log_files(self, input_filenames: List[str], output_filename: str):
        ffi = self._ffi
        lib = self._lib

        n_input_filenames = len(input_filenames)

        input_filenames_alloc = [ffi.new('char[]', f.encode('utf-8')) for f in input_filenames]
        input_filenames_c = ffi.new('char*[]', input_filenames_alloc)
        output_filename_c = ffi.new('char[]', output_filename.encode('utf-8'))

        lib.merge_game_log_files(input_filenames_c, n_input_filenames, output_filename_c)

        for f in input_filenames:
            os.remove(f)

    def restore_data_loader(self, gens: List[Generation], row_counts: List[int],
                            file_sizes: List[int], n_total_rows: int):
        ffi = self._ffi
        lib = self._lib

        assert len(gens) == len(row_counts) == len(file_sizes)

        n = len(gens)
        if n == 0:
            return

        gens_c = ffi.new('int[]', gens)
        row_counts_c = ffi.new('int[]', row_counts)
        file_sizes_c = ffi.new('int64_t[]', file_sizes)
        lib.DataLoader_restore(self._data_loader, n_total_rows, n, gens_c, row_counts_c,
                               file_sizes_c)

    def add_gen(self, gen: Generation, num_rows: int, file_size: int):
        self._lib.DataLoader_add_gen(self._data_loader, gen, num_rows, file_size)

    def create_data_batches(self, minibatch_size: int, n_minibatches: int, window_start: int,
                            window_end: int, target_names: List[str], gen: Generation,
                            apply_symmetry=True) -> Iterable[DataBatch]:
        ffi = self._ffi
        lib = self._lib

        n_samples = minibatch_size * n_minibatches
        n_targets = len(target_names)

        input_shape_info = self.shape_info_dict['input']
        target_shape_infos = [self.shape_info_dict[name] for name in target_names]

        input_shape = input_shape_info.shape
        target_shapes = [info.shape for info in target_shape_infos]
        mask_shapes = [(1, ) for _ in target_shape_infos]

        input_shape_size = math.prod(input_shape)
        target_shape_sizes = [math.prod(shape) for shape in target_shapes]
        mask_shape_sizes = [math.prod(shape) for shape in mask_shapes]

        # we smash the input, targets, and masks into a single output array
        combined_size = input_shape_size + sum(target_shape_sizes) + sum(mask_shape_sizes)
        output_shape = (n_samples, combined_size)
        output_tensor = torch.empty(output_shape, dtype=torch.float32)

        output_values_c = ffi.cast('float*', output_tensor.data_ptr())

        target_indices = [s.target_index for s in target_shape_infos]
        target_indices_c = ffi.new('int[]', target_indices)

        gen_range_tensor = torch.empty(2, dtype=torch.int32)
        gen_range_value_c = ffi.cast('int*', gen_range_tensor.data_ptr())

        lib.DataLoader_load(self._data_loader, window_start, window_end, n_samples, apply_symmetry,
                            n_targets, output_values_c, target_indices_c, gen_range_value_c)

        start_gen = gen_range_tensor[0].item()
        end_gen = gen_range_tensor[1].item()

        if start_gen == end_gen:
            gen_str = f'gen {start_gen}'
        else:
            gen_str = f'gens {start_gen} to {end_gen}'

        window_size = window_end - window_start
        logger.info(
            'Sampling from %s of %s (%.1f%%) positions (%s)' %
            (window_size, window_end, 100. * window_size / window_end, gen_str))

        output_tensor = output_tensor.to(self._cuda_device_str)

        for i in range(n_minibatches):
            start = i * minibatch_size
            end = (i + 1) * minibatch_size
            output_slice = output_tensor[start:end]

            input_slice = output_slice[:, :input_shape_size].reshape(-1, *input_shape)

            c = input_shape_size
            target_slices = []
            mask_slices = []
            for target_shape, target_size in zip(target_shapes, target_shape_sizes):
                target_slice = output_slice[:, c:c + target_size].reshape(-1, *target_shape)
                mask_slice = output_slice[:, c + target_size].type(torch.bool)
                target_slices.append(target_slice)
                mask_slices.append(mask_slice)
                c += target_size + 1

            data_batch = DataBatch(input_tensor=input_slice,
                                   target_tensors=target_slices,
                                   target_masks=mask_slices)
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
                int is_primary;
            };

            struct ShapeInfo* get_shape_info_array(const char* paradigm);

            void free_shape_info_array(struct ShapeInfo* info);

            struct DataLoader* DataLoader_new(const char* data_dir, int64_t memory_budget,
                int num_worker_threads, int num_prefetch_threads, const char* paradigm);

            void DataLoader_delete(struct DataLoader* loader);

            void DataLoader_restore(struct DataLoader* loader, int64_t n_total_rows, int n,
                int* gens, int* row_counts, int64_t* file_sizes);

            void DataLoader_add_gen(struct DataLoader* loader, int gen, int num_rows,
                int64_t file_size);

            void DataLoader_load(struct DataLoader* loader, int64_t window_start,
                int64_t window_end, int n_samples, bool apply_symmetry, int n_targets,
                float* output_data_array, int* target_indices_array, int* gen_range);

            void merge_game_log_files(const char** input_filenames, int n_input_filenames,
                const char* output_filename);

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

        paradigm_str_c = ffi.new('char[]', self._paradigm.value.encode('utf-8'))
        shape_info_arr = lib.get_shape_info_array(paradigm_str_c)

        shape_info_dict = {}
        i = 0
        while True:
            info = shape_info_arr[i]
            if info.name == ffi.NULL:
                break
            name = ffi.string(info.name).decode('utf-8')
            shape = tuple([info.dims[j] for j in range(info.num_dims)])
            shape_info = ShapeInfo(name=name, target_index=info.target_index,
                                   primary=bool(info.is_primary), shape=shape)
            shape_info_dict[name] = shape_info
            logger.debug('ShapeInfo: %s -> %s', name, shape_info)
            i += 1

        lib.free_shape_info_array(shape_info_arr)
        return shape_info_dict
