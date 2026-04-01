#pragma once

#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <concepts>

namespace core {

namespace concepts {

template <typename T, typename Game>
concept NetworkHead = requires(typename T::Tensor& tensor) {
  // Head name, which must match name used in python.
  { util::decay_copy(T::kName) } -> std::same_as<const char*>;

  // A kPerActionBased head is one whose output is defined per action (e.g. policy, action-value).
  // More precisely, the shape of the Tensor is such that the *first* dimension corresponds to the
  // number of actions in the game.
  //
  // The significance of such heads is that:
  //
  // 1. They can be packed based on ActionMask
  // 2. They can be symmetrized via Game::Symmetries::apply()
  { util::decay_copy(T::kPerActionBased) } -> std::same_as<bool>;

  // A kGameResultBased head is one whose output is defined per game result (e.g. value). The shape
  // of the Tensor must exactly match that of Game::Types::GameResultTensor.
  //
  // The significance of this is that we need to left/right-rotate them based on the active seat.
  { util::decay_copy(T::kGameResultBased) } -> std::same_as<bool>;

  // A kWinShareBased head is one whose output is defined per player. More precisely, the shape of
  // the Tensor is such that the *last* dimension corresponds to the number of players in the game.
  //
  // The significance of such heads is that we need to left/right-rotate them based on the active
  // seat.
  //
  // Note that the action-value head is both kPerActionBased and kWinShareBased.
  { util::decay_copy(T::kWinShareBased) } -> std::same_as<bool>;

  typename T::Tensor;
  requires eigen_util::concepts::FTensor<typename T::Tensor>;

  // Performs an in-place transformation of the tensor into a more usable space.
  // For example, this might be a softmax() if the network training uses cross-entropy loss.
  //
  // This concept merely requires that the argument is the Tensor type. In actuality, we require
  // it to accept other forms, like an Eigen::TensorMap<...>. The implementations should be
  // templated to allow this. See NetworkHeads.hpp for examples.
  //
  // For kType == kPolicyBasedHead targets, the tensor will be packed based on ActionMask.
  { T::transform(tensor) };

  // Uniformly initializes the tensor in place. This is used in contexts where we don't have a
  // model (e.g. generation-0 self-play and unit-tests).
  //
  // This concept merely requires that the argument is the Tensor type. In actuality, we require
  // it to accept other forms, like an Eigen::TensorMap<...>. The implementations should be
  // templated to allow this. See NetworkHeads.hpp for examples.
  //
  // For kType == kPolicyBasedHead targets, the tensor will be packed based on ActionMask.
  { T::uniform_init(tensor) };
};

}  // namespace concepts

template <typename Game>
struct _IsNetworkHead {
  template <typename T>
  struct Pred {
    static constexpr bool value = concepts::NetworkHead<T, Game>;
  };
};

namespace concepts {

template <typename TT, typename Game>
concept NetworkHeads = requires {
  typename TT::List;
  requires mp::IsTypeListSatisfying<typename TT::List, _IsNetworkHead<Game>::template Pred>;
};

}  // namespace concepts
}  // namespace core
