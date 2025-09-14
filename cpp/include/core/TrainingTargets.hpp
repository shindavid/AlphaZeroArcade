#pragma once

#include "core/concepts/GameConcept.hpp"

#include <unsupported/Eigen/CXX11/Tensor>

namespace core {

// TargetBase is a base class for all training targets. It provides useful defaults. These defaults
// are only relevant for targets that live in TrainingTargets::PrimaryList, as they are only
// applicable to targets that are predicted by the neural network during game-play.
struct TargetBase {
  // Targets that have kPolicyBased == true are shaped like the policy tensor. The significance of
  // this is that:
  //
  // 1. It can be packed based on ActionMask
  // 2. It can be symmetrized via Game::Symmetries::apply()
  static constexpr bool kPolicyBased = false;

  // Targets that have kValueBased == true are shaped like the value tensor. The significance of
  // this is that we need to left/right-rotate them based on the active seat.
  static constexpr bool kValueBased = false;

   // Some network heads output values in the logit space. This function transforms the values
   // into a more usable space. The default implementation is a no-op.
  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&) {}

  // Used for generation-0 evaluations
  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct PolicyTarget : public TargetBase {
  static constexpr const char* kName = "policy";
  static constexpr bool kPolicyBased = true;

  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

// NOTE: If we add a game where we produce non-logit value predictions, we should modify this to
// allow customization. We will likely need this in single-player games.
template <core::concepts::Game Game>
struct ValueTarget : public TargetBase {
  static constexpr const char* kName = "value";
  static constexpr bool kValueBased = true;

  using Tensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ActionValueTarget : public TargetBase {
  static constexpr const char* kName = "action_value";
  using Tensor = Game::Types::ActionValueTensor;
  using GameLogView = Game::Types::GameLogView;
  static constexpr bool kPolicyBased = true;

  static bool tensorize(const GameLogView& view, Tensor&);

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ValueUncertaintyTarget : public TargetBase {
  static constexpr const char* kName = "value_uncertainty";
  using Tensor = eigen_util::FTensor<Eigen::Sizes<1>>;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct ActionValueUncertaintyTarget : public TargetBase {
  static constexpr const char* kName = "action_value_uncertainty";
  using Tensor = Game::Types::ActionValueTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);

  template <typename Derived>
  static void transform(Eigen::TensorBase<Derived>&);

  template <typename Derived>
  static void uniform_init(Eigen::TensorBase<Derived>&);
};

template <core::concepts::Game Game>
struct OppPolicyTarget : public TargetBase {
  static constexpr const char* kName = "opp_policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;

  static bool tensorize(const GameLogView& view, Tensor&);
};

}  // namespace core

#include "inline/core/TrainingTargets.inl"
