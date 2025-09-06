#pragma once

#include "core/concepts/Game.hpp"

namespace core {

template <core::concepts::Game Game>
struct PolicyTarget {
  static constexpr const char* kName = "policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

template <core::concepts::Game Game>
struct ValueTarget {
  static constexpr const char* kName = "value";
  using Tensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

template <core::concepts::Game Game>
struct ActionValueTarget {
  static constexpr const char* kName = "action_value";
  using Tensor = Game::Types::ActionValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

template <core::concepts::Game Game>
struct OppPolicyTarget {
  static constexpr const char* kName = "opp_policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

}  // namespace core

#include "inline/core/TrainingTargets.inl"
