#pragma once

#include "core/DefaultTransposer.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/SearchParadigm.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinShareEncoding.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputEncoder.hpp"
#include "games/nim/InputFrame.hpp"
#include "games/nim/PolicyEncoding.hpp"
#include "games/nim/Symmetries.hpp"

namespace nim {

namespace alpha0 {

using GameResultEncoding = core::WinShareEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = nim::Game;
  using InputFrame = nim::InputFrame;
  using Symmetries = nim::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = nim::alpha0::TensorEncodings;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using NetworkHeads = nim::alpha0::NetworkHeads;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

}  // namespace alpha0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec>;
};

}  // namespace nim
