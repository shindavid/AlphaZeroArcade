#pragma once

#include "alpha0/Traits.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/InputFrame.hpp"
#include "games/hex/InputTensorizor.hpp"
#include "games/hex/PolicyEncoding.hpp"
#include "games/hex/Symmetries.hpp"
#include "games/hex/Transposer.hpp"
#include "util/MetaProgramming.hpp"

namespace hex::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 8;
};

}  // namespace hex::alpha0

namespace core {

template <>
struct EvalSpec<hex::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = hex::Game;
  using InputFrame = hex::InputFrame;
  using Symmetries = hex::Symmetries;
  using Transposer = hex::Transposer;
  using InputTensorizor = hex::InputTensorizor;
  using PolicyEncoding = hex::PolicyEncoding;
  using TrainingTargets = hex::alpha0::TrainingTargets;
  using NetworkHeads = hex::alpha0::NetworkHeads;
  using MctsConfiguration = hex::alpha0::MctsConfiguration;
};

}  // namespace core

namespace hex {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::Traits<Game>>;
};

}  // namespace hex
