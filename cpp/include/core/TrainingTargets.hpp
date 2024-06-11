#pragma once

#include <util/EigenUtil.hpp>

namespace core {

template<typename Game>
struct PolicyTarget {
  static constexpr const char* kName = "policy";
  using Tensor = typename Game::PolicyTensor;
  using GameLogView = typename Game::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

template<typename Game>
struct ValueTarget {
  static constexpr const char* kName = "value";
  using ValueArray = typename Game::ValueArray;
  using Shape = eigen_util::Shape<ValueArray::RowsAtCompileTime>;
  using Tensor = eigen_util::FTensor<Shape>;
  using GameLogView = typename Game::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

template <typename Game>
struct OppPolicyTarget {
  static constexpr const char* kName = "opp_policy";
  using Tensor = typename Game::PolicyTensor;
  using GameLogView = typename Game::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

}  // namespace core
