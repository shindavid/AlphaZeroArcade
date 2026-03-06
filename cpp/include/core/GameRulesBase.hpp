#pragma once

#include "core/BasicTypes.hpp"
#include "util/EigenUtil.hpp"

namespace core {

template <typename Types>
struct RulesBase {
  using State = Types::State;

  static bool is_chance_mode(core::action_mode_t action_mode) { return false; }

  static Types::ChanceDistribution get_chance_distribution(const State& state) {
    return eigen_util::zeros<typename Types::ChanceDistribution>();
  }

  static void backtrack_state(State& state, const State& prev_state) { state = prev_state; }
};
}  // namespace core
