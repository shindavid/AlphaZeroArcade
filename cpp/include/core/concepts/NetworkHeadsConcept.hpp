#pragma once

#include "core/NetworkHeads.hpp"
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

  // Targets that are of type kPolicyBasedHead are shaped like the policy tensor. The significance
  // of this is that:
  //
  // 1. It can be packed based on ActionMask
  // 2. It can be symmetrized via Game::Symmetries::apply()
  //
  // Targets that are of type kValueBasedHead are shaped like the value tensor. The significance of
  // this is that we need to left/right-rotate them based on the active seat.
  { util::decay_copy(T::kType) } -> std::same_as<NetworkHeadType>;

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
