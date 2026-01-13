#include "beta0/Calculations.hpp"

#include "core/BasicTypes.hpp"
#include "util/CppUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
void Calculations<Traits>::calibrate_priors(core::seat_index_t seat, const LocalPolicyArray& P,
                                            const ValueArray& V, ValueArray& U,
                                            LocalActionValueArray& AV,
                                            const LocalActionValueArray& AU) {
  int n = P.size();
  LocalPolicyArray AVs(n);
  for (int i = 0; i < n; ++i) {
    AVs(i) = AV(i, seat);
  }
  shift_AVs(V[seat], P, AVs);

  for (int i = 0; i < n; ++i) {
    AV(i, seat) = AVs(i);
    static_assert(kNumPlayers == 2, "Assuming 2 players here");
    AV(i, 1 - seat) = 1.0f - AVs(i);
  }

  // Recompute U from adjusted AV and original AU
  auto U_in_mat = AU.matrix();
  auto U_across_mat = (AV.rowwise() - V.transpose()).square().matrix();
  auto U_mat = (U_in_mat + U_across_mat).transpose() * P.matrix();
  U = U_mat.array();
}

// TODO: This implementation conservatively prioritizes correctness over performance by using
// std::log() and math::sigmoid() in favor of faster approximations. Later, we can consider
// switching to faster approximations if needed.
template <search::concepts::Traits Traits>
void Calculations<Traits>::shift_AVs(float V, const LocalPolicyArray& P, LocalPolicyArray& AVs) {
  const int n = P.size();

  // Compute logits once.
  LocalPolicyArray lAVs(n);
  for (int i = 0; i < n; ++i) {
    const float a = AVs[i];
    lAVs[i] = std::log(a / (1.0f - a));  // logit
  }

  // Monotone function:
  //   F(c) = sum_i P[i] * sigmoid(lAVs[i] + c) - V
  // F is strictly increasing in c, so we bracket and do safeguarded Newton.
  float minL = lAVs.minCoeff();
  float maxL = lAVs.maxCoeff();

  // Bracket: make all (lAVs+c) very negative / very positive.
  // 12 is already strongly saturated for sigmoid.
  constexpr float kSat = 12.0f;
  float lo = -maxL - kSat;
  float hi = -minL + kSat;

  float c = 0.0f;
  if (c < lo) c = lo;
  if (c > hi) c = hi;

  constexpr int kIters = 16;      // conservative; usually converges faster
  constexpr float kTolF = 1e-7f;  // in value units (since sum(P)=1)

  for (int it = 0; it < kIters; ++it) {
    float F = 0.0f;
    float dF = 0.0f;

    for (int i = 0; i < n; ++i) {
      const float s = math::sigmoid(lAVs[i] + c);
      const float w = P[i];
      F += w * s;
      dF += w * (s * (1.0f - s));
    }

    const float err = F - V;
    if (std::fabs(err) <= kTolF) break;

    // Maintain bracket (monotone increasing).
    if (err < 0.0f)
      lo = c;
    else
      hi = c;

    // Safeguarded Newton.
    const float c_newton = c - err / (dF + 1e-12f);
    c = (c_newton <= lo || c_newton >= hi) ? (0.5f * (lo + hi)) : c_newton;
  }

  // Apply shift in-place: AVs[i] = sigmoid(lAVs[i] + c)
  for (int i = 0; i < n; ++i) {
    AVs[i] = math::sigmoid(lAVs[i] + c);
  }

  if (IS_DEFINED(DEBUG_BUILD)) {
    // Check that P * AVs = V
    float check = 0.0f;
    for (int i = 0; i < n; ++i) {
      check += P[i] * AVs[i];
    }
    if (std::fabs(check - V) > .01f) {
      LOG_ERROR("calibrate_priors failed:");
      LOG_ERROR("V: {}", V);
      LOG_ERROR("check: {}", check);
      LOG_ERROR("P: {}", fmt::streamed(P.transpose()));
      LOG_ERROR("AVs: {}", fmt::streamed(AVs.transpose()));
      throw util::Exception("calibrate_priors failed");
    }
  }
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
