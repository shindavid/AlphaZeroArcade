#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/MetaProgramming.hpp"

// Contains definitions for network heads.
//
// Each *NetworkHead class corresponds to an output head of the neural network. Each class must
// satisfy the core::concepts::NetworkHead concept.
namespace core {

struct NetworkHeadBase {
  static constexpr bool kPerActionBased = false;
  static constexpr bool kGameResultBased = false;
  static constexpr bool kWinShareBased = false;
};

template <core::concepts::Game Game>
struct PolicyNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "policy";
  static constexpr bool kPerActionBased = true;
  using Tensor = Game::Types::PolicyTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

// NOTE: If we add a game where we produce non-logit value predictions, we should modify this to
// allow customization. We will likely need this in single-player games.
template <core::concepts::Game Game>
struct ValueNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "value";
  static constexpr bool kGameResultBased = true;
  using Tensor = Game::Types::GameResultTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ActionValueNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "action_value";
  static constexpr bool kPerActionBased = true;
  static constexpr bool kWinShareBased = true;
  using Tensor = Game::Types::ActionValueTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ValueUncertaintyNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "value_uncertainty";
  static constexpr bool kWinShareBased = true;

  using Tensor = Game::Types::WinShareTensor;

  // sigmoid is performed on pytorch side, so no transform performed here
  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&) {}

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ActionValueUncertaintyNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "action_value_uncertainty";
  static constexpr bool kPerActionBased = true;
  static constexpr bool kWinShareBased = true;
  using Tensor = Game::Types::ActionValueTensor;

  // sigmoid is performed on pytorch side, so no transform performed here
  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&) {}

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

namespace alpha0 {

template <core::concepts::Game Game>
struct StandardNetworkHeads {
  using PolicyHead = PolicyNetworkHead<Game>;
  using ValueHead = ValueNetworkHead<Game>;
  using ActionValueHead = ActionValueNetworkHead<Game>;

  using List = mp::TypeList<PolicyHead, ValueHead, ActionValueHead>;
};

template <core::concepts::Game Game>
using StandardNetworkHeadsList = typename StandardNetworkHeads<Game>::List;

}  // namespace alpha0

namespace beta0 {

template <core::concepts::Game Game>
struct StandardNetworkHeads {
  using PolicyHead = PolicyNetworkHead<Game>;
  using ValueHead = ValueNetworkHead<Game>;
  using ActionValueHead = ActionValueNetworkHead<Game>;
  using ValueUncertaintyHead = core::ValueUncertaintyNetworkHead<Game>;
  using ActionValueUncertaintyHead = core::ActionValueUncertaintyNetworkHead<Game>;

  using List1 = alpha0::StandardNetworkHeads<Game>::List;
  using List2 = mp::TypeList<ValueUncertaintyHead, ActionValueUncertaintyHead>;
  using List = mp::Concat_t<List1, List2>;
};

template <core::concepts::Game Game>
using StandardNetworkHeadsList = typename StandardNetworkHeads<Game>::List;

}  // namespace beta0

}  // namespace core

#include "inline/core/NetworkHeads.inl"
