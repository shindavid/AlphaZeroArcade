#pragma once

#include "alpha0/GameLogBundle.hpp"  // IWYU pragma: keep
#include "alpha0/PlayerBundle.hpp"   // IWYU pragma: keep
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/SearchParadigm.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossEncoding.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/InputEncoder.hpp"
#include "games/hex/InputFrame.hpp"
#include "games/hex/PolicyEncoding.hpp"
#include "games/hex/Symmetries.hpp"
#include "games/hex/Transposer.hpp"

namespace hex {

namespace alpha0 {

using GameResultEncoding = core::WinLossEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

using TrainingTargets = core::alpha0::StandardTrainingTargets<TensorEncodings>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings, Symmetries>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 8;
};

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = hex::Game;
  using InputFrame = hex::InputFrame;
  using Symmetries = hex::Symmetries;
  using Transposer = hex::Transposer;
  using TensorEncodings = hex::alpha0::TensorEncodings;
  using TrainingTargets = hex::alpha0::TrainingTargets;
  using NetworkHeads = hex::alpha0::NetworkHeads;
  using MctsConfiguration = hex::alpha0::MctsConfiguration;
};

}  // namespace alpha0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec>;
};

}  // namespace hex
