#pragma once

#include "alpha0/Traits.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputEncoder.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/PolicyEncoding.hpp"
#include "games/chess/Symmetries.hpp"
#include "games/chess/Transposer.hpp"
#include "util/MetaProgramming.hpp"

namespace a0achess {

using TensorEncodings = core::TensorEncodings<InputEncoder, PolicyEncoding>;

namespace alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 18;  // 9 moves per player = reasonablish quarter-life
};

}  // namespace alpha0

}  // namespace a0achess

namespace core {

template <>
struct EvalSpec<a0achess::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = a0achess::Game;
  using InputFrame = a0achess::InputFrame;
  using Symmetries = a0achess::Symmetries;
  using Transposer = a0achess::Transposer;
  using TensorEncodings = a0achess::TensorEncodings;
  using TrainingTargets = a0achess::alpha0::TrainingTargets;
  using NetworkHeads = a0achess::alpha0::NetworkHeads;
  using MctsConfiguration = a0achess::alpha0::MctsConfiguration;
};

}  // namespace core

namespace a0achess {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::Traits<Game>>;
};

}  // namespace a0achess
