#pragma once

#include <core/GameLog.hpp>

#define FFI_MACRO(Game)                                                                    \
  using GameReadLog = core::GameReadLog<Game>;                                             \
  using State = Game::State;                                                               \
  using GameTensor = Game::InputTensorizor::Tensor;                                        \
                                                                                           \
  extern "C" {                                                                             \
                                                                                           \
  core::ShapeInfo* get_shape_info_array() { return GameReadLog::get_shape_info_array(); }  \
                                                                                           \
  void free_shape_info_array(core::ShapeInfo* info) { delete[] info; }                     \
                                                                                           \
  GameReadLog* GameLog_new(const char* filename, int game_index) {                         \
    return new GameReadLog(filename, game_index);                                          \
  }                                                                                        \
                                                                                           \
  void GameLog_delete(GameReadLog* log) { delete log; }                                    \
                                                                                           \
  int GameLog_num_sampled_positions(GameReadLog* log) {                                    \
    return log->num_sampled_positions();                                                   \
  }                                                                                        \
                                                                                           \
  void GameLog_replay(GameReadLog* log) { log->replay(); }                                 \
                                                                                           \
  void GameLog_load(GameReadLog* log, int index, bool apply_symmetry, float* input_values, \
                    int* target_indices, float** target_arrays, bool** target_masks) {     \
    log->load(index, apply_symmetry, input_values, target_indices, target_arrays,          \
              target_masks);                                                               \
  }                                                                                        \
                                                                                           \
  void Game_tensorize(State* start, int num_states, float* input_values) {                 \
    auto input = Game::InputTensorizor::tensorize(start, start + (num_states - 1));        \
    std::copy(input.data(), input.data() + input.size(), input_values);                    \
  }                                                                                        \
                                                                                           \
  void init() { Game::static_init(); }                                                     \
                                                                                           \
  }  // extern "C"
