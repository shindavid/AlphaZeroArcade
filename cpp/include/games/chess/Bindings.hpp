#pragma once

#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/TrainingTargets.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace chess {

struct Keys {
  using TransposeKey = uint64_t;
  using EvalKey = Game::State;

  // TODO: hash sequence of states back up to T-50 or last zeroing move, whichever is closer
  static TransposeKey transpose_key(const Game::StateHistory& history) {
    throw std::runtime_error("Not implemented");
  }

  template <typename Iter>
  static EvalKey eval_key(Iter start, Iter cur) {
    return *cur;
  };
};

namespace alpha0 {

struct TrainingTargets {
  using BoardShape = Eigen::Sizes<kBoardDim, kBoardDim>;

  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using PrimaryList = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget>;
  using AuxList = mp::TypeList<OppPolicyTarget>;
};

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 18;  // 9 moves per player = reasonablish quarter-life
};

}  // namespace alpha0

}  // namespace chess

namespace core {

template <>
struct InputTensorizor<chess::Game> : public chess::InputTensorizor {
  using Keys = chess::Keys;
};

template <>
struct EvalSpec<chess::Game, core::kParadigmAlphaZero> {
  using Game = chess::Game;
  using TrainingTargets = chess::alpha0::TrainingTargets;
  using MctsConfiguration = chess::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<chess::Game, core::kParadigmBetaZero> {
  using Game = chess::Game;
  using TrainingTargets = chess::alpha0::TrainingTargets;
  using MctsConfiguration = chess::alpha0::MctsConfiguration;
};

}  // namespace core
