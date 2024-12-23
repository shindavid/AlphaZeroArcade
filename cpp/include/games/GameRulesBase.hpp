#pragma once

#include <core/concepts/Game.hpp>
#include <util/EigenUtil.hpp>

namespace game_base {

template <typename Types>
struct RulesBase {
  static bool is_chance_mode(core::action_mode_t action_mode) { return false; }

  static Types::ChanceDistribution get_chance_distribution(const Types::State& state) {
    return eigen_util::zeros<typename Types::ChanceDistribution>();
  }
};
}  // namespace game_base

