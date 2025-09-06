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

  typename T::Tensor;
  requires eigen_util::concepts::FTensor<typename T::Tensor>;

  // If we have a valid training target, populates tensor_ref and returns true.
  // Otherwise, returns false.
  { T::tensorize(view, tensor_ref) } -> std::same_as<bool>;
};

}  // namespace concepts

template <typename Game>
struct _IsTrainingTarget {
  template <typename T>
  struct Pred {
    static constexpr bool value = concepts::TrainingTarget<T, Game>;
  };
};

namespace concepts {

template <typename TT, typename Game>
concept TrainingTargets = requires {
  typename TT::List;
  requires mp::IsTypeListSatisfying<typename TT::List, _IsTrainingTarget<Game>::template Pred>;
};

}  // namespace concepts
}  // namespace core
