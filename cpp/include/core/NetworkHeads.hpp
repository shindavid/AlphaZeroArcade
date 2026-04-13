#pragma once

#include "core/concepts/GameConcept.hpp"
#include "core/concepts/TensorEncodingsConcept.hpp"
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

template <core::concepts::TensorEncodings TensorEncodings>
struct PolicyNetworkHead : public NetworkHeadBase {
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  static constexpr char kName[] = "policy";
  static constexpr bool kPerActionBased = true;
  using Tensor = PolicyEncoding::Tensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

// NOTE: If we add a game where we produce non-logit value predictions, we should modify this to
// allow customization. We will likely need this in single-player games.
template <core::concepts::TensorEncodings TensorEncodings>
struct ValueNetworkHead : public NetworkHeadBase {
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  static constexpr char kName[] = "value";
  static constexpr bool kGameResultBased = true;
  using Tensor = GameResultEncoding::Tensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::TensorEncodings TensorEncodings>
struct ActionValueNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "action_value";
  static constexpr bool kPerActionBased = true;
  static constexpr bool kWinShareBased = true;
  using Tensor = TensorEncodings::ActionValueEncoding::Tensor;

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::TensorEncodings TensorEncodings>
struct ValueUncertaintyNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "value_uncertainty";
  static constexpr bool kWinShareBased = true;

  using Tensor = TensorEncodings::WinShareTensor;

  // sigmoid is performed on pytorch side, so no transform performed here
  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&) {}

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::TensorEncodings TensorEncodings>
struct ActionValueUncertaintyNetworkHead : public NetworkHeadBase {
  static constexpr char kName[] = "action_value_uncertainty";
  static constexpr bool kPerActionBased = true;
  static constexpr bool kWinShareBased = true;
  using Tensor = TensorEncodings::ActionValueEncoding::Tensor;

  // sigmoid is performed on pytorch side, so no transform performed here
  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&) {}

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

namespace alpha0 {

template <core::concepts::TensorEncodings TensorEncodings>
struct StandardNetworkHeads {
  using Game = TensorEncodings::Game;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using PolicyHead = PolicyNetworkHead<TensorEncodings>;
  using ValueHead = ValueNetworkHead<TensorEncodings>;
  using ActionValueHead = ActionValueNetworkHead<TensorEncodings>;

  using List = mp::TypeList<PolicyHead, ValueHead, ActionValueHead>;
};

template <core::concepts::TensorEncodings TensorEncodings>
using StandardNetworkHeadsList = typename StandardNetworkHeads<TensorEncodings>::List;

}  // namespace alpha0

}  // namespace core

#include "inline/core/NetworkHeads.inl"
