#include <core/GameLog.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/Tensorizor.hpp>

using GameReadLog = core::GameReadLog<c4::GameState, c4::Tensorizor>;

extern "C" {

GameReadLog* c4_GameReadLog_new(const char* filename) { return new GameReadLog(filename); }

void c4_GameReadLog_delete(GameReadLog* log) { delete log; }

void c4_GameReadLog_load(GameReadLog* log, int index, bool apply_symmetry, float* input,
                         const char** keys, float** values, int num_keys) {
  log->load(index, apply_symmetry, input, keys, values, num_keys);
}

}  // extern "C"
