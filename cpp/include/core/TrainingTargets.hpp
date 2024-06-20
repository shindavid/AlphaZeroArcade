#pragma once

#include <util/EigenUtil.hpp>

namespace core {

template<typename Game>
struct PolicyTarget {
  static constexpr const char* kName = "policy";
  using Tensor = typename Game::Types::PolicyTensor;
  using GameLogView = typename Game::Types::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

template<typename Game>
struct ValueTarget {
  static constexpr const char* kName = "value";
  using ValueArray = typename Game::Types::ValueArray;
  using Shape = eigen_util::Shape<eigen_util::extract_length_v<ValueArray>>;
  using Tensor = eigen_util::FTensor<Shape>;
  using GameLogView = typename Game::Types::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

template <typename Game>
struct OppPolicyTarget {
  static constexpr const char* kName = "opp_policy";
  using Tensor = typename Game::Types::PolicyTensor;
  using GameLogView = typename Game::Types::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

}  // namespace core

#include <inline/core/TrainingTargets.inl>
