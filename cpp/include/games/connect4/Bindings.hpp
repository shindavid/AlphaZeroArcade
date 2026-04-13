#pragma once

#include "core/DefaultTransposer.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/SearchParadigm.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossDrawEncoding.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputEncoder.hpp"
#include "games/connect4/InputFrame.hpp"
#include "games/connect4/PolicyEncoding.hpp"
#include "games/connect4/Symmetries.hpp"

namespace c4 {

namespace alpha0 {

using GameResultEncoding = core::WinLossDrawEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings, Symmetries>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 10.583;  // likely too big, just keeping previous value
};

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = c4::Game;
  using InputFrame = c4::InputFrame;
  using Symmetries = c4::Symmetries;
  using Transposer = core::DefaultTransposer<Game, InputFrame>;
  using TensorEncodings = c4::alpha0::TensorEncodings;
  using TrainingTargets = c4::alpha0::TrainingTargets;
  using NetworkHeads = c4::alpha0::NetworkHeads;
  using MctsConfiguration = c4::alpha0::MctsConfiguration;
};

}  // namespace alpha0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec>;
};

}  // namespace c4
