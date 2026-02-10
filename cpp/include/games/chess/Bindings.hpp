#pragma once

#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputTensorizor.hpp"

namespace chess {

struct Keys {
  using TransposeKey = uint64_t;
  using EvalKey = Game::State::zobrist_hash_t;
  using InputTensorizor = core::InputTensorizor<Game>;

  // TODO: hash sequence of states back up to T-50 or last zeroing move, whichever is closer
  static TransposeKey transpose_key(const Game::State& state) {
    throw std::runtime_error("Not implemented");
  }

  static EvalKey eval_key(InputTensorizor* input_tensorizor) {
    throw std::runtime_error("Not implemented");
  }
};

namespace alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

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
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = chess::Game;
  using TrainingTargets = chess::alpha0::TrainingTargets;
  using NetworkHeads = chess::alpha0::NetworkHeads;
  using MctsConfiguration = chess::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<chess::Game, core::kParadigmBetaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmBetaZero;
  using Game = chess::Game;
  using TrainingTargets = chess::alpha0::TrainingTargets;
  using NetworkHeads = chess::alpha0::NetworkHeads;
  using MctsConfiguration = chess::alpha0::MctsConfiguration;
};

}  // namespace core
