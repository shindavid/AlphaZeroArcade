#pragma once

#include "core/BasicTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/Gaussian1D.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
struct Calculations {
  using Game = Traits::Game;
  using LogitValueArray = Game::Types::LogitValueArray;
  using ValueArray = Game::Types::ValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  // P, V, U, AV, AU come from the neural network.
  //
  // If the network is accurate, these should satisfy:
  //
  // * V = sum_i P[i] * AV[i] (Law of Total Expectation)
  // * U = sum_i P[i] * (AU[i] + AV[i]^2) - V^2 (Law of Total Variance)
  //
  // However, in practice these equalities do not hold exactly.
  //
  // This function rewrites AV and U in place to ensure these equalities hold. It also populates
  // lAU and lAV with the logit-value belief mean/variance pairs corresponding to the adjusted AV
  // and AU.
  //
  // The rewriting is done by applying a constant shift to the logit-value beliefs underlying AV:
  //
  //  AV[i] -> sigmoid(logit(AV[i]) + c)
  //
  // We solve for c such that the first equality holds.
  //
  // We cannot do the same for AU due to the structure of the Law of Variance, so we instead
  // recompute U from the adjusted AV and the original AU.
  static void calibrate_priors(core::seat_index_t seat, const LocalPolicyArray& P,
                               const ValueArray& V, ValueArray& U, LocalActionValueArray& AV,
                               const LocalActionValueArray& AU, LocalActionValueArray& lAV,
                               LocalActionValueArray& lAU);

  // Solves for a constant c that satisfies:
  //
  // sum_i P[i] * sigmoid(lAVs[i] + c) = V
  static float monotone_solve(float V, const LocalPolicyArray& P, const LocalPolicyArray& lAVs);

  static void populate_logit_value_beliefs(const ValueArray& Q, const ValueArray& W,
                                           LogitValueArray& lQW);
  static util::Gaussian1D compute_logit_value_belief(float Q, float W);
};

}  // namespace beta0

#include "inline/beta0/Calculations.inl"
