#include <core/GameLog.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/Tensorizor.hpp>

using GameReadLog = core::GameReadLog<c4::GameState, c4::Tensorizor>;

extern "C" {

GameReadLog* c4_GameReadLog_new(const char* filename) { return new GameReadLog(filename); }

void c4_GameReadLog_delete(GameReadLog* log) { delete log; }

void c4_load(GameReadLog* log, int index, bool apply_symmetry, float* input, float* policy,
             float* value, const char** aux_keys, float** aux, int num_aux) {
  log->load(index, apply_symmetry, input, policy, value, aux_keys, aux, num_aux);
}

// c4::GameState* c4_GameState_new() {
//   return new c4::GameState();
// }

// void c4_GameState_delete(c4::GameState* state) { delete state; }

// void c4_GameState_load(c4::GameState* state, const char* filename, size_t index) {
//   state->load(filename, index);
// }

// c4::Tensorizor* c4_Tensorizor_new() { return new c4::Tensorizor(); }

// void c4_Tensorizor_delete(c4::Tensorizor* tensorizor) { delete tensorizor; }

// void c4_Tensorizor_tensorize(const c4::Tensorizor* tensorizor, bool* tensor,
//                              const c4::GameState* state) {
//   tensorizor->tensorize(tensor, *state);
// }

}  // extern "C"
