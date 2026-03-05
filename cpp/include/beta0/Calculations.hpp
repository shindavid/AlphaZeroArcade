#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/Math.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct Calculations {
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using LogitValueArray = Game::Types::LogitValueArray;
  using ValueArray = Game::Types::ValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;

  using Array1D = LocalPolicyArray;
  using Array2D = LocalActionValueArray;
  using Mask = eigen_util::DArray<kMaxBranchingFactor, bool>;

  static constexpr float kPiSquaredOver3 = math::kPi * math::kPi / 3.0f;

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
  static void p2l(const Array1D& AV, const Array1D& AU, Array1D& lAV);

  // Converts from logit space to prob-space
  //
  // mu_p = sigmoid(mu_l / sqrt(1 + pi^2 * s_l / 3))
  //
  // s_p = s_l * sigmoid(mu_l)^2 * (1 - sigmoid(mu_l))^2
  static void l2p(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU);
  static void l2p_fast(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU);
  static void l2p(const Array2D& lAV, const Array2D& lAU, Array2D& AV);
  static void l2p_fast(const Array2D& lAV, const Array2D& lAU, Array2D& AV);

  template<typename Derived>
  static void l2p(const Array1D& lAV, const Array1D& lAU, Eigen::ArrayBase<Derived>& AV);
  static void l2p_fast(const Array1D& lAV, const Array1D& lAU, Array1D& AV);

  static ValueArray scale_uncertainty(const ValueArray& V, const ValueArray& U01);
  static LocalActionValueArray scale_uncertainty(const LocalActionValueArray& AV,
                                                 const LocalActionValueArray& AU01);

  // Computes and returns a constant beta such that:
  //
  // V = P * l2p(p2l(AV) + beta * sqrt(AU))
  static float compute_beta(core::seat_index_t seat, const LocalPolicyArray& P, const ValueArray& V,
                            const LocalActionValueArray& AV, const LocalActionValueArray& AU);

  // Q_dot_product() does a 1D evaluation for seat, using exact_dot_product(). Then, if
  // kNumPlayers == 2, it exactly sets out[1-seat] to 1.0f - out[seat].
  static void Q_dot_product(core::seat_index_t seat, const Array1D& Q, const Array1D& pi,
                            ValueArray& out);

  // W_dot_product() does a 1D evaluation for seat, using exact_dot_product(). Then, if
  // kNumPlayers == 2, it exactly sets out[1-seat] to out[seat].
  static void W_dot_product(const Array1D& W, const Array1D& pi, ValueArray& out);

  // When X is constant c over the support of pi, naively doing out = X^T * pi yields a value that
  // is slightly off from c, which causes some issues downstream. The exact_dot_product() function
  // detects that case and perform exact overwrites instead.
  //
  // Note that the order of arguments is important for exactness: X must be the first argument and
  // pi the second, since we check for constancy in X and check for zero-support in pi.
  static float exact_dot_product(const Array1D& X, const Array1D& pi);

 private:
  template <typename LogitFn>
  static void p2l_helper(const Array2D& AV, const Array2D& AU, Array2D& lAV, Array2D& lAU,
                         LogitFn&& logit_fn);

  template <typename LogitFn>
  static void p2l_helper(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW,
                         LogitFn&& logit_fn);

  template <typename LogitFn>
  static void p2l_helper(const Array1D& AV, const Array1D& AU, Array1D& lAV, LogitFn&& logit_fn);

  template <typename SigmoidFn>
  static void l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU,
                         SigmoidFn&& sigmoid_fn);

  template <typename SigmoidFn>
  static void l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV,
                         SigmoidFn&& sigmoid_fn);
};

}  // namespace beta0

#include "inline/beta0/Calculations.inl"
