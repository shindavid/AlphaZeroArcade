#pragma once

#include "core/ActionValueEncoding.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game_, typename InputEncoder_, typename PolicyEncoding_,
          typename GameResultEncoding_>
struct TensorEncodings {
  using Game = Game_;
  using InputEncoder = InputEncoder_;
  using PolicyEncoding = PolicyEncoding_;
  using GameResultEncoding = GameResultEncoding_;
  using ActionValueEncoding = core::ActionValueEncoding<Game, PolicyEncoding>;
  using WinShareTensor = eigen_util::FTensor<Eigen::Sizes<Game::Constants::kNumPlayers>>;
};

}  // namespace core
