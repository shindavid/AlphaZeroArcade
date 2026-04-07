#pragma once

#include "alpha0/SearchSpec.hpp"
#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinShareEncoding.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputEncoder.hpp"
#include "games/stochastic_nim/InputFrame.hpp"
#include "games/stochastic_nim/PolicyEncoding.hpp"
#include "games/stochastic_nim/Symmetries.hpp"
#include "util/MetaProgramming.hpp"

namespace stochastic_nim {

using GameResultEncoding = core::WinShareEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

}  // namespace stochastic_nim

namespace stochastic_nim::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

}  // namespace stochastic_nim::alpha0

namespace core {

template <>
struct EvalSpec<stochastic_nim::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = stochastic_nim::Game;
  using InputFrame = stochastic_nim::InputFrame;
  using Symmetries = stochastic_nim::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = stochastic_nim::TensorEncodings;
  using TrainingTargets = stochastic_nim::alpha0::TrainingTargets;
  using NetworkHeads = stochastic_nim::alpha0::NetworkHeads;
  using MctsConfiguration = stochastic_nim::alpha0::MctsConfiguration;
};

}  // namespace core

namespace stochastic_nim {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::SearchSpec<Game>>;
};

}  // namespace stochastic_nim
