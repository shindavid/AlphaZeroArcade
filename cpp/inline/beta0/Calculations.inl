#include "beta0/Calculations.hpp"

#include "core/BasicTypes.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
void Calculations<Traits>::calibrate_priors(core::seat_index_t seat, const LocalPolicyArray& P,
                                            const ValueArray& V, ValueArray& U,
                                            LocalActionValueArray& AV,
                                            const LocalActionValueArray& AU,
                                            LocalActionValueArray& lAV,
                                            LocalActionValueArray& lAU) {
  int n = P.size();
  LocalPolicyArray lAVs(n);
  for (int i = 0; i < n; ++i) {
    LogitValueArray child_lAUV;
    populate_logit_value_beliefs(AV.row(i), AU.row(i), child_lAUV);
    for (int p = 0; p < kNumPlayers; ++p) {
      lAV(i, p) = child_lAUV[p].mean();
      lAU(i, p) = child_lAUV[p].variance();
    }
    lAVs(i) = lAV(i, seat);
  }
  float c = monotone_solve(V[seat], P, lAVs);
  for (int i = 0; i < n; ++i) {
    float shifted_lAVs = lAVs(i) + c;
    lAV(i, seat) = shifted_lAVs;

    static_assert(kNumPlayers == 2, "Assuming 2 players here");
    lAV(i, 1 - seat) = -shifted_lAVs;

    AV(i, seat) = math::fast_coarse_sigmoid(shifted_lAVs);
    AV(i, 1 - seat) = 1.0f - AV(i, seat);
  }

  // Recompute U from adjusted AV and original AU
  auto U_in_mat = AU.matrix();
  auto U_across_mat = (AV.rowwise() - V.transpose()).square().matrix();
  auto U_mat = (U_in_mat + U_across_mat).transpose() * P.matrix();
  U = U_mat.array();
}

template <search::concepts::Traits Traits>
float Calculations<Traits>::monotone_solve(float V, const LocalPolicyArray& P,
                                           const LocalPolicyArray& lAVs) {
  const int n = P.size();

  const float maxL = lAVs.maxCoeff();
  const float minL = lAVs.minCoeff();

  // Bracket using "sigmoid saturation" constant.
  // 12 is already very saturated (sigmoid(±12) ~ 6e-6 / 0.999994).
  // You can tune this based on your LUT's clamping behavior.
  constexpr float kSat = 12.0f;
  float lo = -maxL - kSat;
  float hi = -minL + kSat;

  // Start at 0 clamped into the bracket (often a good guess).
  float c = 0.0f;
  if (c < lo) c = lo;
  if (c > hi) c = hi;

  const float* pP = P.data();
  const float* pL = lAVs.data();

  // Fixed iterations: typically enough for single-precision policy work.
  // 6–8 is a good sweet spot.
  constexpr int kIters = 8;

  for (int it = 0; it < kIters; ++it) {
    float F = 0.0f;
    float dF = 0.0f;

    // Accumulate F(c) and F'(c)
    for (int i = 0; i < n; ++i) {
      const float s = math::fast_coarse_sigmoid(pL[i] + c);
      const float w = pP[i];
      F += w * s;
      dF += w * (s * (1.0f - s));
    }

    const float err = F - V;

    // Maintain bracket (monotone F).
    if (err < 0.0f) {
      lo = c;
    } else {
      hi = c;
    }

    if (dF <= 1e-12f) {
      c = 0.5f * (lo + hi);
      continue;
    }

    // Safeguarded Newton.
    // dF should be > 0 unless everything is numerically saturated.
    const float c_newton = c - err / dF;

    // If Newton step leaves the bracket, fall back to bisection.
    if (c_newton <= lo || c_newton >= hi) {
      c = 0.5f * (lo + hi);
    } else {
      c = c_newton;
    }
  }

  return c;
}

template <search::concepts::Traits Traits>
void Calculations<Traits>::populate_logit_value_beliefs(const ValueArray& Q, const ValueArray& W,
                                                        LogitValueArray& lQW) {
  if (kNumPlayers == 2) {
    // In this case, we only need to compute for one player, since the other is just negation.
    lQW[0] = compute_logit_value_belief(Q[0], W[0]);
    lQW[1] = -lQW[0];
  } else {
    for (core::seat_index_t p = 0; p < kNumPlayers; ++p) {
      lQW[p] = compute_logit_value_belief(Q[p], W[p]);
    }
  }
}

template <search::concepts::Traits Traits>
util::Gaussian1D Calculations<Traits>::compute_logit_value_belief(float Q, float W) {
  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;
  constexpr float kWidth = kMax - kMin;
  constexpr float kInvWidth = 1.0f / kWidth;

  if (Q <= kMin) {
    return util::Gaussian1D::neg_inf();
  } else if (Q >= kMax) {
    return util::Gaussian1D::pos_inf();
  }
  if (W == 0) {
    float theta = math::fast_coarse_logit((Q - kMin) * kInvWidth);
    return util::Gaussian1D(theta, 0.f);
  }

  float mu = Q;
  float sigma_sq = W;

  // Rescale Q and W to reflect [0, 1] range
  mu = (mu - kMin) * kInvWidth;
  sigma_sq *= kInvWidth * kInvWidth;

  float mult = 1.0f / (mu * mu * (1 - mu) * (1 - mu));

  float theta1 = math::fast_coarse_logit(mu);
  float theta2 = (0.5 - mu) * sigma_sq * mult;
  float theta = theta1 - theta2;

  float omega_sq = sigma_sq * mult;
  return util::Gaussian1D(theta, omega_sq);
}

}  // namespace beta0
