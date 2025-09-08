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
  { util::decay_copy(T::kName) } -> std::same_as<const char*>;
  { util::decay_copy(T::kValueBased) } -> std::same_as<bool>;
  { util::decay_copy(T::kPolicyBased) } -> std::same_as<bool>;
  { util::decay_copy(T::kUsesLogitScale) } -> std::same_as<bool>;

  typename T::Tensor;
  requires eigen_util::concepts::FTensor<typename T::Tensor>;

  // If we have a valid training target, populates tensor_ref and returns true.
  // Otherwise, returns false.
  { T::tensorize(view, tensor_ref) } -> std::same_as<bool>;
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
