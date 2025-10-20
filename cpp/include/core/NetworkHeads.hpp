#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <cstdint>

// Contains definitions for network heads.
//
// Each *NetworkHead class corresponds to an output head of the neural network. Each class must
// satisfy the core::concepts::NetworkHead concept.
namespace core {

enum NetworkHeadType : int8_t {
  // Targets that are of type kPolicyBasedHead are shaped like the policy tensor. The significance
  // of this is that:
  //
  // 1. It can be packed based on ActionMask
  // 2. It can be symmetrized via Game::Symmetries::apply()
  kPolicyBasedHead,

  // Targets that are of type kValueBasedHead are shaped like the value tensor. The significance of
  // this is that we need to left/right-rotate them based on the active seat.
  //
  // This can differ from kWinShareBasedHead in games where the value head predicts something
  // other than the win share for each player. For example, in a 2-player game, the value head might
  // predict W/L/D, which has shape (3,), while the win share tensor has shape (2,).
  kValueBasedHead,

  // Targets that are of type kWinSharedBasedHead are shaped like the win share tensor. The
  // significance of this is that we need to left/right-rotate them based on the active seat.
  //
  // This can differ from kValueBasedHead in games where the value head predicts something
  // other than the win share for each player. For example, in a 2-player game, the value head might
  // predict W/L/D, which has shape (3,), while the win share tensor has shape (2,).
  kWinShareBasedHead,

  // Targets that are of type kDefaultHeadType are neither policy-based nor value-based. This is the
  // default type for targets that do not fit into either of the above categories.
  kDefaultHeadType
};

template <core::concepts::Game Game>
struct PolicyNetworkHead {
  static constexpr const char* kName = "policy";
  static constexpr NetworkHeadType kType = NetworkHeadType::kPolicyBasedHead;
  using Tensor = Game::Types::PolicyTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

// NOTE: If we add a game where we produce non-logit value predictions, we should modify this to
// allow customization. We will likely need this in single-player games.
template <core::concepts::Game Game>
struct ValueNetworkHead {
  static constexpr const char* kName = "value";
  static constexpr NetworkHeadType kType = NetworkHeadType::kValueBasedHead;
  using Tensor = Game::Types::GameResultTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ActionValueNetworkHead {
  static constexpr const char* kName = "action_value";
  static constexpr NetworkHeadType kType = NetworkHeadType::kPolicyBasedHead;
  using Tensor = Game::Types::ActionValueTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ValueUncertaintyNetworkHead {
  static constexpr const char* kName = "value_uncertainty";
  static constexpr NetworkHeadType kType = NetworkHeadType::kWinShareBasedHead;

  using Tensor = Game::Types::WinShareTensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&) {}

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ActionValueUncertaintyNetworkHead {
  static constexpr const char* kName = "action_value_uncertainty";
  static constexpr NetworkHeadType kType = NetworkHeadType::kPolicyBasedHead;
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
