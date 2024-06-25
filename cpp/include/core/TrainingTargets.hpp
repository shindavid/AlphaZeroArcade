#pragma once

#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

namespace core {

namespace concepts {

template <typename T, typename GameLogView>
concept TrainingTarget = requires (const GameLogView& view) {
  { util::decay_copy(T::kName) } -> std::same_as<const char*>;
  requires eigen_util::concepts::FTensor<typename T::Tensor>;
  { T::tensorize(view) } -> std::same_as<typename T::Tensor>;
};

}  // namespace concepts

template<typename GameLogView>
struct IsTrainingTarget {
  template<typename T>
  struct Pred {
    static constexpr bool value = concepts::TrainingTarget<T, GameLogView>;
  };
};

namespace concepts {

template <typename T, typename GameLogView>
concept TrainingTargetList = mp::IsTypeListSatisfying<T, IsTrainingTarget<GameLogView>::template Pred>;

}  // namespace concepts

template<typename Game>
struct PolicyTarget {
  static constexpr const char* kName = "policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

template<typename Game>
struct ValueTarget {
  static constexpr const char* kName = "value";
  using ValueArray = Game::Types::ValueArray;
  using Shape = eigen_util::Shape<eigen_util::extract_length_v<ValueArray>>;
  using Tensor = eigen_util::FTensor<Shape>;
  using GameLogView = Game::Types::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

template <typename Game>
struct OppPolicyTarget {
  static constexpr const char* kName = "opp_policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static Tensor tensorize(const GameLogView& view);
};

}  // namespace core

#include <inline/core/TrainingTargets.inl>
