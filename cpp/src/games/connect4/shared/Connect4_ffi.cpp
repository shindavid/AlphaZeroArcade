#include <core/GameLog.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/Tensorizor.hpp>

using GameReadLog = core::GameReadLog<c4::GameState, c4::Tensorizor>;

extern "C" {

core::ShapeInfo* get_shape_info_array() { return GameReadLog::get_shape_info(); }

void free_shape_info_array(core::ShapeInfo* info) { delete[] info; }

GameReadLog* GameReadLog_new(const char* filename) { return new GameReadLog(filename); }

void GameReadLog_delete(GameReadLog* log) { delete log; }

void GameReadLog_load(GameReadLog* log, int index, bool apply_symmetry, const char** keys,
                      float** values, int num_keys) {
  log->load(index, apply_symmetry, keys, values, num_keys);
}

}  // extern "C"
