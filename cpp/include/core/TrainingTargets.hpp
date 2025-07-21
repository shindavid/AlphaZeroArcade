#pragma once

#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

namespace core {

namespace concepts {

template <typename T, typename GameLogView>
concept TrainingTarget =
  requires(const GameLogView& view, typename T::Tensor& tensor_ref, seat_index_t active_seat) {
    { util::decay_copy(T::kName) } -> std::same_as<const char*>;
    requires eigen_util::concepts::FTensor<typename T::Tensor>;

    // If we have a valid training target, populates tensor_ref and returns true.
    // Otherwise, returns false.
    { T::tensorize(view, tensor_ref) } -> std::same_as<bool>;
  };

}  // namespace concepts

template <typename GameLogView>
struct IsTrainingTarget {
  template <typename T>
  struct Pred {
    static constexpr bool value = concepts::TrainingTarget<T, GameLogView>;
  };
};

namespace concepts {

template <typename T, typename GameLogView>
concept TrainingTargetList =
  mp::IsTypeListSatisfying<T, IsTrainingTarget<GameLogView>::template Pred>;

}  // namespace concepts

template <typename Game>
struct PolicyTarget {
  static constexpr const char* kName = "policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

template <typename Game>
struct ValueTarget {
  static constexpr const char* kName = "value";
  using Tensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

template <typename Game>
struct ActionValueTarget {
  static constexpr const char* kName = "action_value";
  using Tensor = Game::Types::ActionValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

template <typename Game>
struct OppPolicyTarget {
  static constexpr const char* kName = "opp_policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

}  // namespace core

#include "inline/core/TrainingTargets.inl"
