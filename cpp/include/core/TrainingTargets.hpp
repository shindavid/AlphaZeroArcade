#pragma once

#include <core/BasicTypes.hpp>
#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

#include <string>

namespace core {

namespace concepts {

template <typename T, typename GameLogView>
concept TrainingTarget = requires (const GameLogView& view) {
  requires eigen_util::concepts::FTensor<typename T::Tensor>;
  { T::name() } -> std::same_as<std::string>;
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

template<typename Game, action_type_t ActionType=0>
struct PolicyTarget {
  using Tensor = mp::TypeAt_t<typename Game::Types::Policy, ActionType>;
  using GameLogView = Game::Types::GameLogView;

  static std::string name() { return util::create_string("policy%d", ActionType); }
  static Tensor tensorize(const GameLogView& view);
};

template <typename Game>
struct ValueTarget {
  using Tensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static std::string name() { return "value"; }
  static Tensor tensorize(const GameLogView& view);
};

template<typename Game, action_type_t ActionType=0>
struct ActionValueTarget {
  using Tensor = mp::TypeAt_t<typename Game::Types::ActionValues, ActionType>;
  using GameLogView = Game::Types::GameLogView;

  static std::string name() { return util::create_string("policy%d", ActionType); }
  static Tensor tensorize(const GameLogView& view);
};

template <typename Game, action_type_t ActionType=0>
struct OppPolicyTarget {
  using Tensor = mp::TypeAt_t<typename Game::Types::Policy, ActionType>;
  using GameLogView = Game::Types::GameLogView;

  static std::string name() { return util::create_string("opp_policy%d", ActionType); }
  static Tensor tensorize(const GameLogView& view);
};

}  // namespace core

#include <inline/core/TrainingTargets.inl>
