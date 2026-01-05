#pragma once

#include "search/concepts/TraitsConcept.hpp"
#include "util/Gaussian1D.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
struct Calculations {
  using Game = Traits::Game;
  using LogitValueArray = Game::Types::LogitValueArray;
  using ValueArray = Game::Types::ValueArray;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  static void populate_logit_value_beliefs(const ValueArray& Q, const ValueArray& W,
                                           LogitValueArray& lQW);
  static util::Gaussian1D compute_logit_value_belief(float Q, float W);
};

}  // namespace beta0

#include "inline/beta0/Calculations.inl"
