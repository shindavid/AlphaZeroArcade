#include "beta0/Calculations.hpp"

#include "core/BasicTypes.hpp"
#include "util/CppUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

namespace beta0 {

template <core::concepts::Game Game>
void Calculations<Game>::calibrate_priors(core::seat_index_t seat, const LocalPolicyArray& P,
                                          ValueArray& V, ValueArray& U,
                                          LocalActionValueArray& AV,
                                          const LocalActionValueArray& AU) {
  constexpr float kMin = Game::GameResults::kMinValue + 1e-6f;
  constexpr float kMax = Game::GameResults::kMaxValue - 1e-6f;
  for (int p = 0; p < kNumPlayers; ++p) {
    V(p) = std::clamp(V(p), kMin, kMax);
  }
  int n = P.size();
  LocalPolicyArray AVs(n);
  for (int i = 0; i < n; ++i) {
    AVs(i) = std::clamp(AV(i, seat), kMin, kMax);
  }
  shift_AVs(V[seat], P, AVs);

  for (int i = 0; i < n; ++i) {
    float x = std::clamp(AVs(i), kMin, kMax);
    AV(i, seat) = x;
    static_assert(kNumPlayers == 2, "Assuming 2 players here");
    AV(i, 1 - seat) = 1.0f - x;
  }

  // Recompute U from adjusted AV and original AU
  auto U_in_mat = AU.matrix();
  auto U_across_mat = (AV.rowwise() - V.transpose()).square().matrix();
  dot_product(U_in_mat + U_across_mat, P, U);
}

// TODO: This implementation conservatively prioritizes correctness over performance by using
// std::log() and math::sigmoid() in favor of faster approximations. Later, we can consider
// switching to faster approximations if needed.
template <core::concepts::Game Game>
void Calculations<Game>::shift_AVs(float V, const LocalPolicyArray& P, LocalPolicyArray& AVs) {
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
    bool out_of_bounds = (c_newton < lo) || (c_newton > hi);
    c = out_of_bounds ? (0.5f * (lo + hi)) : c_newton;
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

template <core::concepts::Game Game>
void Calculations<Game>::populate_logit_value_beliefs(const ValueArray& Q, const ValueArray& W,
                                                      LogitValueArray& lQW,
                                                      ComputationCheckMethod method) {
  if (kNumPlayers == 2) {
    // In this case, we only need to compute for one player, since the other is just negation.
    lQW[0] = compute_logit_value_belief(Q[0], W[0], method);
    lQW[1] = -lQW[0];
    if (lQW[1].variance() < 0.f) {
      RELEASE_ASSERT(W[1] == 0.f, "Q: [{}, {}] W: [{}, {}]", Q[0], Q[1], W[0], W[1]);
      RELEASE_ASSERT(Q[1] == 0.f || Q[1] == 1.f, "Q: [{}, {}] W: [{}, {}]", Q[0], Q[1], W[0], W[1]);
    }
  } else {
    for (core::seat_index_t p = 0; p < kNumPlayers; ++p) {
      lQW[p] = compute_logit_value_belief(Q[p], W[p], method);
    }
  }
}

template <core::concepts::Game Game>
util::Gaussian1D Calculations<Game>::compute_logit_value_belief(float Q, float W,
                                                                ComputationCheckMethod method) {
  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;
  constexpr float kWidth = kMax - kMin;
  constexpr float kInvWidth = 1.0f / kWidth;

  if (Q <= kMin) {
    RELEASE_ASSERT(method != kAssertFinite, "(Q, W): ({}, {})", Q, W);
    return util::Gaussian1D::neg_inf();
  } else if (Q >= kMax) {
    RELEASE_ASSERT(method != kAssertFinite, "(Q, W): ({}, {})", Q, W);
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

  mu = std::clamp(mu, 1e-6f, 1.0f - 1e-6f);
  sigma_sq = std::max(sigma_sq, 1e-6f);

  float mult = 1.0f / (mu * mu * (1 - mu) * (1 - mu));

  float theta1 = math::fast_coarse_logit(mu);
  float theta2 = (0.5 - mu) * sigma_sq * mult;
  float theta = theta1 - theta2;

  float omega_sq = sigma_sq * mult;
  RELEASE_ASSERT(omega_sq > 0.f, "(Q, W, theta, omega_sq): ({}, {}, {}, {})", Q, W, theta,
                 omega_sq);
  return util::Gaussian1D(theta, omega_sq);
}

template <core::concepts::Game Game>
void Calculations<Game>::dot_product(const LocalActionValueArray& X, const LocalPolicyArray& pi,
                                     ValueArray& out) {
  out = (X.matrix().transpose() * pi.matrix()).array();

  const auto support = (pi != 0.0f);
  const bool full_support = support.all();

  for (int p = 0; p < kNumPlayers; ++p) {
    const auto xcol = X.col(p);

    float x_min;
    float x_max;
    if (full_support) {
      x_min = xcol.minCoeff();
      x_max = xcol.maxCoeff();
    } else {
      x_min = support.select(xcol, std::numeric_limits<float>::max()).minCoeff();
      x_max = support.select(xcol, std::numeric_limits<float>::lowest()).maxCoeff();
    }

    if (x_min == x_max) {
      out(p) = x_min;
    } else {
      out(p) = std::clamp(out(p), 1e-6f, 1.0f - 1e-6f);  // avoid extreme values
    }
  }
}

}  // namespace beta0
