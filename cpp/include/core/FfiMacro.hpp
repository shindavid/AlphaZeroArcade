#pragma once

#include <core/GameLog.hpp>

#define FFI_MACRO(Game)                                                                    \
  using GameLog = core::GameLog<Game>;                                                     \
  using State = Game::State;                                                               \
  using GameTensor = Game::InputTensorizor::Tensor;                                        \
                                                                                           \
  extern "C" {                                                                             \
                                                                                           \
  core::ShapeInfo* get_shape_info_array() { return GameLog::get_shape_info_array(); }      \
                                                                                           \
  void free_shape_info_array(core::ShapeInfo* info) { delete[] info; }                     \
                                                                                           \
  GameLog* GameLog_new(const char* filename) { return new GameLog(filename); }             \
                                                                                           \
  void GameLog_delete(GameLog* log) { delete log; }                                        \
                                                                                           \
  int GameLog_num_sampled_positions(GameLog* log) { return log->num_sampled_positions(); } \
                                                                                           \
  void GameLog_replay(GameLog* log) { log->replay(); }                                     \
                                                                                           \
  void GameLog_load(GameLog* log, int index, bool apply_symmetry, float* input_values,     \
                    int* target_indices, float** target_arrays, bool** target_masks) {     \
    log->load(index, apply_symmetry, input_values, target_indices, target_arrays,          \
              target_masks);                                                               \
  }                                                                                        \
                                                                                           \
  void Game_tensorize(State* start_state, int num_states, float* input_values) {                 \
    auto input = Game::InputTensorizor::tensorize(start_state, start_state + (num_states - 1));  \
    std::copy(input.data(), input.data() + input.size(), input_values);                          \
  }                                                                                        \
                                                                                           \
  void init() { Game::static_init(); }                                                     \
                                                                                           \
  }  // extern "C"
