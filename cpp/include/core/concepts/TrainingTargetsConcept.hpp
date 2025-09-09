#pragma once

#include "core/BasicTypes.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <concepts>

namespace core {

namespace concepts {

template <typename T, typename Game>
concept TrainingTarget = requires(const typename Game::Types::GameLogView& view,
                                  typename T::Tensor& tensor_ref, seat_index_t active_seat) {
  // Name of the target. Must match name used in python.
  { util::decay_copy(T::kName) } -> std::same_as<const char*>;

  // Targets that have kPolicyBased == true are shaped like the policy tensor. The significance of
  // this is that:
  //
  // 1. It can be packed based on ActionMask
  // 2. It can be symmetrized via Game::Symmetries::apply()
  //
  // For TrainingTarget classes that inherit from core::TargetBase, this is false by default.
  { util::decay_copy(T::kPolicyBased) } -> std::same_as<bool>;

  // Targets that have kValueBased == true are shaped like the value tensor. The significance of
  // this is that we need to left/right-rotate them based on the active seat.
  //
  // For TrainingTarget classes that inherit from core::TargetBase, this is false by default.
  { util::decay_copy(T::kValueBased) } -> std::same_as<bool>;

  typename T::Tensor;
  requires eigen_util::concepts::FTensor<typename T::Tensor>;

  // If we have a valid training target, populates tensor_ref and returns true.
  // Otherwise, returns false.
  { T::tensorize(view, tensor_ref) } -> std::same_as<bool>;

  // Performs an in-place transformation of the tensor into a more usable space.
  // For example, this might be a softmax() if the network training uses cross-entropy loss.
  //
  // For TrainingTarget classes that inherit from core::TargetBase, this is a no-op by default.
  //
  // This concept merely requires that the argument is the Tensor type. In actuality, we require
  // it to accept other forms, like an Eigen::TensorMap<...>. The implementations should be
  // templated to allow this. See core::TrainingTargets.inl for examples.
  //
  // For kPolicyBased == true targets, the tensor will be packed based on ActionMask.
  { T::transform(tensor_ref) };

  // Uniformly initializes the tensor in place. This is used in contexts where we don't have a
  // model (e.g. generation-0 self-play and unit-tests).
  //
  // This only needs to be defined for targets that are in TrainingTargets::PrimaryList.
  //
  // This concept merely requires that the argument is the Tensor type. In actuality, we require
  // it to accept other forms, like an Eigen::TensorMap<...>. The implementations should be
  // templated to allow this. See core::TrainingTargets.inl for examples.
  //
  // For kPolicyBased == true targets, the tensor will be packed based on ActionMask.
  { T::uniform_init(tensor_ref) };
};

}  // namespace concepts

template <typename Game>
struct _IsTarget {
  template <typename T>
  struct Pred {
    static constexpr bool value = concepts::TrainingTarget<T, Game>;
  };
};

namespace concepts {

template <typename TT, typename Game>
concept TrainingTargets = requires {
  // TT::PrimaryList consists of targets that will be predicted by the neural net during game-play.
  //
  // TT::AuxList consists of targets that will only be predicted during training. These targets will
  // be stripped out of the neural net before it is exported for game-play.

  typename TT::PrimaryList;
  typename TT::AuxList;

  requires mp::IsTypeListSatisfying<typename TT::PrimaryList, _IsTarget<Game>::template Pred>;
  requires mp::IsTypeListSatisfying<typename TT::AuxList, _IsTarget<Game>::template Pred>;
};

}  // namespace concepts
}  // namespace core
