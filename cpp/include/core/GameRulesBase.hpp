#pragma once

#include "core/BasicTypes.hpp"
#include "core/RulesResult.hpp"

namespace core {

template <typename Types>
struct RulesBase {
  using State = Types::State;
  using Result = core::RulesResult<Types>;

  static constexpr game_phase_t get_game_phase(const State&) { return 0; }
  static constexpr bool is_chance_phase(game_phase_t) { return false; }

  static void backtrack_state(State& state, const State& prev_state) { state = prev_state; }
};
}  // namespace core
