#pragma once

#include "alpha0/GameLogBundle.hpp"  // IWYU pragma: keep
#include "alpha0/PlayerBundle.hpp"   // IWYU pragma: keep
#include "alpha0/GameLogBundle.hpp"  // IWYU pragma: keep
#include "alpha0/PlayerBundle.hpp"   // IWYU pragma: keep
#include "beta0/GameLogBundle.hpp"  // IWYU pragma: keep
#include "beta0/PlayerBundle.hpp"   // IWYU pragma: keep
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

using GameResultEncoding = core::WinLossDrawEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 10.583;  // likely too big, just keeping previous value
};

namespace alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings, Symmetries>;

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  static constexpr const char* kName = "alpha0";
  using Game = c4::Game;
  using InputFrame = c4::InputFrame;
  using Symmetries = c4::Symmetries;
  using Transposer = core::DefaultTransposer<Game, InputFrame>;
  using TensorEncodings = c4::TensorEncodings;
  using TrainingTargets = c4::alpha0::TrainingTargets;
  using NetworkHeads = c4::alpha0::NetworkHeads;
  using MctsConfiguration = c4::MctsConfiguration;
};

}  // namespace alpha0

namespace beta0 {

using TrainingTargets = core::beta0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::beta0::StandardNetworkHeads<TensorEncodings, Symmetries>;

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmBetaZero;
  static constexpr const char* kName = "beta0";
  using Game = c4::Game;
  using InputFrame = c4::InputFrame;
  using Symmetries = c4::Symmetries;
  using Transposer = core::DefaultTransposer<Game, InputFrame>;
  using TensorEncodings = c4::TensorEncodings;
  using TrainingTargets = c4::beta0::TrainingTargets;
  using NetworkHeads = c4::beta0::NetworkHeads;
  using MctsConfiguration = c4::MctsConfiguration;
};

}  // namespace beta0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec, beta0::Spec>;
};

}  // namespace c4
