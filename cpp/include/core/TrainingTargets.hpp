#pragma once

#include "core/concepts/GameConcept.hpp"

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

  // If kUsesLogitScale is true, then the neural network makes predictions for this target in the
  // logit space.
  static constexpr bool kUsesLogitScale = false;

  // Used for generation-0 evaluations
  template <typename ActionMask, typename Tensor>
  static void uniform_init(const ActionMask&, Tensor&);
};

template <core::concepts::Game Game>
struct PolicyTarget : public TargetBase {
  static constexpr const char* kName = "policy";
  using Tensor = Game::Types::PolicyTensor;
  using GameLogView = Game::Types::GameLogView;
  using ActionMask = Game::Types::ActionMask;
  static constexpr bool kPolicyBased = true;
  static constexpr bool kUsesLogitScale = true;

  static bool tensorize(const GameLogView& view, Tensor&);

  template<typename Dst>
  static void uniform_init(const ActionMask&, Dst&);
};

// NOTE: If we add a game where we produce non-logit value predictions, we should modify this to
// allow customization. We will likely need this in single-player games.
template <core::concepts::Game Game>
struct ValueTarget : public TargetBase {
  static constexpr const char* kName = "value";
  using Tensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;
  using ActionMask = Game::Types::ActionMask;
  static constexpr bool kValueBased = true;
  static constexpr bool kUsesLogitScale = true;

  static bool tensorize(const GameLogView& view, Tensor&);

  template<typename Dst>
  static void uniform_init(const ActionMask&, Dst&);
};

template <core::concepts::Game Game>
struct ActionValueTarget : public TargetBase {
  static constexpr const char* kName = "action_value";
  using Tensor = Game::Types::ActionValueTensor;
  using GameLogView = Game::Types::GameLogView;
  using ActionMask = Game::Types::ActionMask;
  static constexpr bool kPolicyBased = true;

  static bool tensorize(const GameLogView& view, Tensor&);

  template<typename Dst>
  static void uniform_init(const ActionMask&, Dst&);
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
