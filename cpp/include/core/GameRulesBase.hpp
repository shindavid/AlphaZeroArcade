#pragma once

#include "core/BasicTypes.hpp"
#include "util/EigenUtil.hpp"
#include "core/RulesResult.hpp"

namespace core {

template <typename Types, typename Derived>
struct RulesBase {
  using State = Types::State;
  using Result = core::RulesResult<Types>;

  static bool is_chance_mode(core::action_mode_t action_mode) { return false; }

  static Types::ChanceDistribution get_chance_distribution(const State& state) {
    return eigen_util::zeros<typename Types::ChanceDistribution>();
  }

  static void backtrack_state(State& state, const State& prev_state) { state = prev_state; }

  static Types::ActionMask get_legal_moves(const State& state) {
    return Derived::analyze(state, core::MoveInfo{-1, -1}).valid_actions();
  }
};
}  // namespace core
