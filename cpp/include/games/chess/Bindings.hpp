#pragma once

#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/Symmetries.hpp"
#include "core/TrainingTargets.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputTensorizor.hpp"
#include "games/chess/Symmetries.hpp"

namespace a0achess {

namespace alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 18;  // 9 moves per player = reasonablish quarter-life
};

}  // namespace alpha0

}  // namespace a0achess

namespace core {

template <> struct Symmetries<a0achess::Game> : public a0achess::Symmetries {};
template <> struct InputTensorizor<a0achess::Game> : public a0achess::InputTensorizor {};

template <>
struct EvalSpec<a0achess::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = a0achess::Game;
  using TrainingTargets = a0achess::alpha0::TrainingTargets;
  using NetworkHeads = a0achess::alpha0::NetworkHeads;
  using MctsConfiguration = a0achess::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<a0achess::Game, core::kParadigmBetaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmBetaZero;
  using Game = a0achess::Game;
  using TrainingTargets = a0achess::alpha0::TrainingTargets;
  using NetworkHeads = a0achess::alpha0::NetworkHeads;
  using MctsConfiguration = a0achess::alpha0::MctsConfiguration;
};

}  // namespace core

#include "inline/games/chess/Bindings.inl"
