#pragma once

#include "alpha0/SearchSpec.hpp"
#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinShareEncoding.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputEncoder.hpp"
#include "games/nim/InputFrame.hpp"
#include "games/nim/PolicyEncoding.hpp"
#include "games/nim/Symmetries.hpp"
#include "util/MetaProgramming.hpp"

namespace nim {

using GameResultEncoding = core::WinShareEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

}  // namespace nim

namespace nim::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

}  // namespace nim::alpha0

namespace core {

template <>
struct EvalSpec<nim::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = nim::Game;
  using InputFrame = nim::InputFrame;
  using Symmetries = nim::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = nim::TensorEncodings;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using NetworkHeads = nim::alpha0::NetworkHeads;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

}  // namespace core

namespace nim {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::SearchSpec<Game>>;
};

}  // namespace nim
