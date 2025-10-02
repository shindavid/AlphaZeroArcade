#pragma once

#include "alphazero/Traits.hpp"
#include "betazero/Traits.hpp"
#include "core/Constants.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/DataLoader.hpp"
#include "search/GameLog.hpp"
#include "util/Exceptions.hpp"

#ifdef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must not be defined for game-ffi's");
#endif

namespace detail {

template <core::concepts::Game Game>
struct FfiFunctions {
  using AlphaZeroTraits = alpha0::Traits<Game>;
  using AlphaZeroGameReadLog = search::GameReadLog<AlphaZeroTraits>;
  using AlphaZeroDataLoader = search::DataLoader<AlphaZeroTraits>;

  using BetaZeroTraits = beta0::Traits<Game>;
  using BetaZeroGameReadLog = search::GameReadLog<BetaZeroTraits>;
  using BetaZeroDataLoader = search::DataLoader<BetaZeroTraits>;

  using DataLoader = search::DataLoaderBase;

  static DataLoader* DataLoader_new(const char* data_dir, int64_t memory_budget,
                                    int num_worker_threads, int num_prefetch_threads,
                                    const char* paradigm) {
    DataLoader::Params params{data_dir, memory_budget, num_worker_threads, num_prefetch_threads};
    core::SearchParadigm p = core::parse_search_paradigm(paradigm);
    switch (p) {
      case core::kParadigmAlphaZero:
        return new AlphaZeroDataLoader(params);
      case core::kParadigmBetaZero:
        return new BetaZeroDataLoader(params);
      default:
        throw util::Exception("Unknown search paradigm '{}'", paradigm);
    }
  }

  static void DataLoader_delete(DataLoader* loader) { delete loader; }

  static void DataLoader_restore(DataLoader* loader, int64_t n_total_rows, int n, int* gens,
                                 int* row_counts, int64_t* file_sizes) {
    DataLoader::RestoreParams params{n_total_rows, n, gens, row_counts, file_sizes};
    loader->restore(params);
  }

  static void DataLoader_add_gen(DataLoader* loader, int gen, int num_rows, int64_t file_size) {
    DataLoader::AddGenParams params{gen, num_rows, file_size};
    loader->add_gen(params);
  }

  static void DataLoader_load(DataLoader* loader, int64_t window_start, int64_t window_end,
                              int n_samples, bool apply_symmetry, int n_targets,
                              float* output_data_array, int* target_indices_array, int* gen_range,
                              int* version_check) {
    DataLoader::LoadParams params{window_start,         window_end, n_samples,
                                  apply_symmetry,       n_targets,  output_data_array,
                                  target_indices_array, gen_range,  version_check};
    loader->load(params);
  }

  static void merge_game_log_files(const char** input_filenames, int n_input_filenames,
                                   const char* output_filename) {
    search::GameLogCommon::merge_files(input_filenames, n_input_filenames, output_filename);
  }

  static search::ShapeInfo* get_input_shapes(const char* paradigm) {
    core::SearchParadigm p = core::parse_search_paradigm(paradigm);
    switch (p) {
      case core::kParadigmAlphaZero:
        return AlphaZeroGameReadLog::get_input_shapes();
      case core::kParadigmBetaZero:
        return BetaZeroGameReadLog::get_input_shapes();
      default:
        throw util::Exception("Unknown search paradigm '{}'", paradigm);
    }
  }

  static search::ShapeInfo* get_target_shapes(const char* paradigm) {
    core::SearchParadigm p = core::parse_search_paradigm(paradigm);
    switch (p) {
      case core::kParadigmAlphaZero:
        return AlphaZeroGameReadLog::get_target_shapes();
      case core::kParadigmBetaZero:
        return BetaZeroGameReadLog::get_target_shapes();
      default:
        throw util::Exception("Unknown search paradigm '{}'", paradigm);
    }
  }

  static search::ShapeInfo* get_head_shapes(const char* paradigm) {
    core::SearchParadigm p = core::parse_search_paradigm(paradigm);
    switch (p) {
      case core::kParadigmAlphaZero:
        return AlphaZeroGameReadLog::get_head_shapes();
      case core::kParadigmBetaZero:
        return BetaZeroGameReadLog::get_head_shapes();
      default:
        throw util::Exception("Unknown search paradigm '{}'", paradigm);
    }
  }

  static void free_shape_info_array(search::ShapeInfo* info) { delete[] info; }

  static void init() { Game::static_init(); }
};

}  // namespace detail

#define FFI_MACRO(Game)                                                                           \
  using FfiFunctions = detail::FfiFunctions<Game>;                                                \
  using DataLoader = FfiFunctions::DataLoader;                                                    \
                                                                                                  \
  extern "C" {                                                                                    \
                                                                                                  \
  DataLoader* DataLoader_new(const char* data_dir, int64_t memory_budget, int num_worker_threads, \
                             int num_prefetch_threads, const char* paradigm) {                    \
    return FfiFunctions::DataLoader_new(data_dir, memory_budget, num_worker_threads,              \
                                        num_prefetch_threads, paradigm);                          \
  }                                                                                               \
                                                                                                  \
  void DataLoader_delete(DataLoader* loader) { FfiFunctions::DataLoader_delete(loader); }         \
                                                                                                  \
  void DataLoader_restore(DataLoader* loader, int64_t n_total_rows, int n, int* gens,             \
                          int* row_counts, int64_t* file_sizes) {                                 \
    FfiFunctions::DataLoader_restore(loader, n_total_rows, n, gens, row_counts, file_sizes);      \
  }                                                                                               \
                                                                                                  \
  void DataLoader_add_gen(DataLoader* loader, int gen, int num_rows, int64_t file_size) {         \
    FfiFunctions::DataLoader_add_gen(loader, gen, num_rows, file_size);                           \
  }                                                                                               \
                                                                                                  \
  void DataLoader_load(DataLoader* loader, int64_t window_start, int64_t window_end,              \
                       int n_samples, bool apply_symmetry, int n_targets,                         \
                       float* output_data_array, int* target_indices_array, int* gen_range,       \
                       int* version_check) {                                                      \
    return FfiFunctions::DataLoader_load(loader, window_start, window_end, n_samples,             \
                                         apply_symmetry, n_targets, output_data_array,            \
                                         target_indices_array, gen_range, version_check);         \
  }                                                                                               \
                                                                                                  \
  void merge_game_log_files(const char** input_filenames, int n_input_filenames,                  \
                            const char* output_filename) {                                        \
    FfiFunctions::merge_game_log_files(input_filenames, n_input_filenames, output_filename);      \
  }                                                                                               \
                                                                                                  \
  search::ShapeInfo* get_input_shapes(const char* paradigm) {                                     \
    return FfiFunctions::get_input_shapes(paradigm);                                              \
  }                                                                                               \
                                                                                                  \
  search::ShapeInfo* get_target_shapes(const char* paradigm) {                                    \
    return FfiFunctions::get_target_shapes(paradigm);                                             \
  }                                                                                               \
                                                                                                  \
  search::ShapeInfo* get_head_shapes(const char* paradigm) {                                      \
    return FfiFunctions::get_head_shapes(paradigm);                                               \
  }                                                                                               \
                                                                                                  \
  void free_shape_info_array(search::ShapeInfo* info) {                                           \
    FfiFunctions::free_shape_info_array(info);                                                    \
  }                                                                                               \
                                                                                                  \
  void init() { FfiFunctions::init(); }                                                           \
                                                                                                  \
  }  // extern "C"
