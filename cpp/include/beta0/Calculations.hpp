#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/Math.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct Calculations {
  using LogitValueArray = Game::Types::LogitValueArray;
  using ValueArray = Game::Types::ValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;

  using Array1D = LocalPolicyArray;
  using Array2D = LocalActionValueArray;

  static constexpr float kPiSquaredOver3 = math::kPi * math::kPi / 3.0f;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static_assert(kNumPlayers <= 2, "Only 2-player games supported for now.");

  // Current Calculations implementation assumes game results are in [0,1]
  static_assert(Game::GameResults::kMinValue == 0.f);
  static_assert(Game::GameResults::kMaxValue == 1.f);

  // Converts from prob-space to logit space
  //
  // mu_l = logit(mu_p) + s_p * (mu_p - 0.5) / (mu_p^2 * (1 - mu_p)^2)
  //
  // s_l = s_p / (mu_p^2 * (1 - mu_p)^2)
  static void p2l(const Array2D& AV, const Array2D& AU, Array2D& lAV, Array2D& lAU);
  static void p2l_fast(const Array2D& AV, const Array2D& AU, Array2D& lAV, Array2D& lAU);
  static void p2l(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW);
  static void p2l_fast(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW);

  // Converts from logit space to prob-space
  //
  // mu_p = sigmoid(mu_l / sqrt(1 + pi^2 * s_l / 3))
  //
  // s_p = s_l * sigmoid(mu_l)^2 * (1 - sigmoid(mu_l))^2
  static void l2p(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU);
  static void l2p_fast(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU);
  static void l2p(const Array2D& lAV, const Array2D& lAU, Array2D& AV);
  static void l2p_fast(const Array2D& lAV, const Array2D& lAU, Array2D& AV);
  static void l2p(const Array1D& lAV, const Array1D& lAU, Array1D& AV);
  static void l2p_fast(const Array1D& lAV, const Array1D& lAU, Array1D& AV);

  // Computes and returns a constant beta such that:
  //
  // V = P * l2p(lAV + beta, lAU)
  static float compute_beta(const LocalPolicyArray& P, const ValueArray& V,
                            const LocalActionValueArray& lAV, const LocalActionValueArray& lAU);

  // Naively doing out = X^T * pi can lead to numerical precision issues when X is constant
  // on the support of pi. This function detects that case and does an exact overwrite instead.
  static void dot_product(const LocalActionValueArray& X, const LocalPolicyArray& pi,
                          ValueArray& out);

 private:
  template <typename LogitFn>
  static void p2l_helper(const Array2D& AV, const Array2D& AU, Array2D& lAV, Array2D& lAU,
                         LogitFn&& logit_fn);

  template <typename LogitFn>
  static void p2l_helper(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW,
                         LogitFn&& logit_fn);

  template <typename SigmoidFn>
  static void l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU,
                         SigmoidFn&& sigmoid_fn);

  template <typename SigmoidFn>
  static void l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV,
                         SigmoidFn&& sigmoid_fn);

};

}  // namespace beta0

#include "inline/beta0/Calculations.inl"
