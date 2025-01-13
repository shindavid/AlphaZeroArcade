#include <core/FfiMacro.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <games/stochastic_nim/players/PerfectPlayer.hpp>

FFI_MACRO(stochastic_nim::Game);

extern "C" {
State* Game_new_state(int stones_left, int mode) {
  return new State(stones_left, stochastic_nim::kPlayer0, mode);
}

using PerfectStrategy = stochastic_nim::PerfectStrategy;

PerfectStrategy* PerfectStrategy_new() { return new PerfectStrategy(); }

float get_state_value_before(PerfectStrategy* strategy, int stones_left) {
  return strategy->get_state_value_before(stones_left);
}

float get_state_value_after(PerfectStrategy* strategy, int stones_left) {
  return strategy->get_state_value_after(stones_left);
}

int get_optimal_action(PerfectStrategy* strategy, int stones_left) {
  return strategy->get_optimal_action(stones_left);
}
} // extern "C"
