#pragma once

#include "alpha0/SearchSpec.hpp"
#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossDrawEncoding.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputEncoder.hpp"
#include "games/connect4/InputFrame.hpp"
#include "games/connect4/PolicyEncoding.hpp"
#include "games/connect4/Symmetries.hpp"
#include "util/MetaProgramming.hpp"

namespace c4 {

using GameResultEncoding = core::WinLossDrawEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

}  // namespace c4

namespace c4::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 10.583;  // likely too big, just keeping previous value
};

}  // namespace c4::alpha0

namespace core {

template <>
struct EvalSpec<c4::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = c4::Game;
  using InputFrame = c4::InputFrame;
  using Symmetries = c4::Symmetries;
  using Transposer = core::DefaultTransposer<Game, InputFrame>;
  using TensorEncodings = c4::TensorEncodings;
  using TrainingTargets = c4::alpha0::TrainingTargets;
  using NetworkHeads = c4::alpha0::NetworkHeads;
  using MctsConfiguration = c4::alpha0::MctsConfiguration;
};

}  // namespace core

namespace c4 {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::SearchSpec<Game>>;
};

}  // namespace c4
