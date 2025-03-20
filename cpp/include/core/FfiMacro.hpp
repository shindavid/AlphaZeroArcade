#pragma once

#include <core/GameLog.hpp>
#include <core/DataLoader.hpp>

#define FFI_MACRO(Game)                                                                    \
  using GameReadLog = core::GameReadLog<Game>;                                             \
  using State = Game::State;                                                               \
  using GameTensor = Game::InputTensorizor::Tensor;                                        \
  using DataLoader = core::DataLoader<Game>;                                               \
  using DataLoaderParams = DataLoader::Params;                                             \
                                                                                           \
  extern "C" {                                                                             \
                                                                                           \
  DataLoader* DataLoader_new(const char* data_dir, int64_t memory_budget,                  \
                             int num_worker_threads, int num_prefetch_threads) {           \
    DataLoaderParams params{data_dir, memory_budget, num_worker_threads,                   \
                                          num_prefetch_threads};                           \
    return new DataLoader(params);                                                         \
  }                                                                                        \
                                                                                           \
  void DataLoader_delete(DataLoader* loader) { delete loader; }                            \
                                                                                           \
  void DataLoader_restore(DataLoader* loader, int n, int* gens, int* row_counts,           \
                          int64_t* file_sizes) {                                           \
    loader->restore(n, gens, row_counts, file_sizes);                                      \
  }                                                                                        \
                                                                                           \
  void DataLoader_add_gen(DataLoader* loader, int gen, int num_rows, int64_t file_size) {  \
    loader->add_gen(gen, num_rows, file_size);                                             \
  }                                                                                        \
                                                                                           \
  void DataLoader_load(DataLoader* loader, int64_t window_size, int n_samples,             \
                       bool apply_symmetry, float* input_data_array,                       \
                       int* target_indices_array, float** target_data_arrays,              \
                       bool** target_mask_arrays, int* start_gen) {                        \
    loader->load(window_size, n_samples, apply_symmetry, input_data_array,                 \
      target_indices_array, target_data_arrays, target_mask_arrays, start_gen);            \
  }                                                                                        \
                                                                                           \
  core::ShapeInfo* get_shape_info_array() { return GameReadLog::get_shape_info_array(); }  \
                                                                                           \
  void free_shape_info_array(core::ShapeInfo* info) { delete[] info; }                     \
                                                                                           \
  void init() { Game::static_init(); }                                                     \
                                                                                           \
  }  // extern "C"
