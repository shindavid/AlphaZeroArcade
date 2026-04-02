#pragma once

#include "core/BasicTypes.hpp"
#include "core/RulesResult.hpp"
#include "core/TrivialChanceDistribution.hpp"
#include "util/Exceptions.hpp"

namespace core {

template <typename Types>
struct RulesBase {
  using State = Types::State;
  using Move = Types::Move;
  using Result = core::RulesResult<Types>;
  using TrivialChanceDistribution = core::TrivialChanceDistribution<Move>;

  static constexpr game_phase_t get_game_phase(const State&) { return 0; }
  static constexpr bool is_chance_phase(game_phase_t) { return false; }
  static TrivialChanceDistribution get_chance_distribution(const State& state) {
    throw util::Exception("Chance distribution not implemented for this game");
  }

  static void backtrack_state(State& state, const State& prev_state) { state = prev_state; }
};
}  // namespace core
