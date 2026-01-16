#pragma once

#include "beta0/Constants.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/Gaussian1D.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct Calculations {
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
  // This function rewrites AV and U in place to ensure these equalities hold.
  //
  // The rewriting is done by applying a constant shift to the logit-value beliefs underlying AV:
  //
  //  AV[i] -> sigmoid(logit(AV[i]) + c)
  //
  // We solve for c such that the first equality holds.
  //
  // We cannot do the same for AU due to the structure of the Law of Variance, so we instead
  // recompute U from the adjusted AV and the original AU.
  static void calibrate_priors(core::seat_index_t seat, const LocalPolicyArray& P, ValueArray& V,
                               ValueArray& U, LocalActionValueArray& AV,
                               const LocalActionValueArray& AU);

  // Replaces AVs with sigmoid(logit(AVs) + c) where c is chosen so that sum_i P[i] * AVs[i] = V
  static void shift_AVs(float V, const LocalPolicyArray& P, LocalPolicyArray& AVs);

  static void populate_logit_value_beliefs(const ValueArray& Q, const ValueArray& W,
                                           LogitValueArray& lQW,
                                           ComputationCheckMethod method = kAssertFinite);

  static util::Gaussian1D compute_logit_value_belief(float Q, float W,
                                                     ComputationCheckMethod method = kAssertFinite);

  // Naively doing out = X^T * pi can lead to numerical precision issues when X is constant
  // on the support of pi. This function detects that case and does an exact overwrite instead.
  static void dot_product(const LocalActionValueArray& X, const LocalPolicyArray& pi,
                          ValueArray& out);
};

}  // namespace beta0

#include "inline/beta0/Calculations.inl"
