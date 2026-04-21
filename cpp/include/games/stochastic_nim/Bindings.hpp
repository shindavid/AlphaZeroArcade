#pragma once

#include "alpha0/GameLogBundle.hpp"  // IWYU pragma: keep
#include "alpha0/PlayerBundle.hpp"   // IWYU pragma: keep
#include "core/DefaultTransposer.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/SearchParadigm.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinShareEncoding.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputEncoder.hpp"
#include "games/stochastic_nim/InputFrame.hpp"
#include "games/stochastic_nim/PolicyEncoding.hpp"
#include "games/stochastic_nim/Symmetries.hpp"

namespace stochastic_nim {

namespace alpha0 {

using GameResultEncoding = core::WinShareEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings, Symmetries>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  static constexpr const char* kName = "alpha0";
  using Game = stochastic_nim::Game;
  using InputFrame = stochastic_nim::InputFrame;
  using Symmetries = stochastic_nim::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = stochastic_nim::alpha0::TensorEncodings;
  using TrainingTargets = stochastic_nim::alpha0::TrainingTargets;
  using NetworkHeads = stochastic_nim::alpha0::NetworkHeads;
  using MctsConfiguration = stochastic_nim::alpha0::MctsConfiguration;
};

}  // namespace alpha0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec>;
};

}  // namespace stochastic_nim
