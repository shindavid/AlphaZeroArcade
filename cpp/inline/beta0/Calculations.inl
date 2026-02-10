#include "beta0/Calculations.hpp"

#include "beta0/Constants.hpp"
#include "util/Asserts.hpp"
#include "util/EigenUtil.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

namespace beta0 {

template <core::concepts::Game Game>
void Calculations<Game>::beta_delta_update(int i, float lambda, float alpha, const Array1D& P,
                                           const Array1D& Q, const Array1D& lAV, float beta_0,
                                           float& beta, Array1D& delta) {

  const float P_sum = P.sum();
  if (!(P_sum > 0.0f)) return;

  // Keep alpha in (0,1) for numerical stability.
  alpha = std::clamp(alpha, 1e-6f, 1.0f - 1e-6f);

  constexpr float c = kBeta;
  const float c1 = c;
  const float c2 = c * c;

  const float lam_d = lambda * (1.0f - alpha);
  const float lam_b = lambda * alpha;

  // s = beta + lAV + delta
  const Array1D S = (beta + lAV + delta);

  // u = sigmoid(c * s)
  const Array1D U = eigen_util::sigmoid(c1 * S);

  // w = u(1-u)
  const Array1D W = U * (1.0f - U);

  // r = u - Q
  const Array1D R = U - Q;

  // IMPORTANT: scale the CE gradient+curvature contributions in s-space:
  //   grad term uses c * r
  //   hess term uses c^2 * w
  const Array1D PW = P * (c2 * W);   // P_k * c^2 * w_k
  const Array1D PR = P * (c1 * R);   // P_k * c   * r_k
  const Array1D PD = P * delta;      // unchanged

  const float Pi  = P[i];
  const float PWi = PW[i];
  const float PRi = PR[i];
  const float Di  = delta[i];

  const float PW_sum = PW.sum();
  const float PR_sum = PR.sum();
  const float PD_sum = PD.sum();

  const float PWnot = PW_sum - PWi;
  const float PRnot = PR_sum - PRi;
  const float PDnot = PD_sum - PD[i];
  const float Pnot  = P_sum - Pi;

  const float b = beta - beta_0;

  const float denY = PWi   + 2.0f * lam_d * Pi;
  const float denZ = PWnot + 2.0f * lam_d * Pnot;

  const bool hasY = (Pi   > 0.0f) && (denY > 0.0f);
  const bool hasZ = (Pnot > 0.0f) && (denZ > 0.0f);

  const float inv_denY = hasY ? (1.0f / denY) : 0.0f;
  const float inv_denZ = hasZ ? (1.0f / denZ) : 0.0f;

  float C = (PWi + PWnot + 2.0f * lam_b * P_sum);
  float B = (PR_sum       + 2.0f * lam_b * P_sum * b);

  if (hasY) {
    C -= (PWi * PWi) * inv_denY;
    B -= (PWi * (PRi + 2.0f * lam_d * Pi * Di)) * inv_denY;
  }
  if (hasZ) {
    C -= (PWnot * PWnot) * inv_denZ;
    B -= (PWnot * (PRnot + 2.0f * lam_d * PDnot)) * inv_denZ;
  }

  if (!(std::fabs(C) > 1e-12f)) return;

  const float x = -B / C;

  float y = 0.0f;
  float z = 0.0f;

  if (hasY) {
    y = -(PWi * x + (PRi + 2.0f * lam_d * Pi * Di)) * inv_denY;
  }
  if (hasZ) {
    z = -(PWnot * x + (PRnot + 2.0f * lam_d * PDnot)) * inv_denZ;
  }

  beta += x;

  // Apply z to all, then fix-up i so net change at i is +y.
  if (z != 0.0f) delta += z;
  delta[i] += (y - z);
}

template <core::concepts::Game Game>
void Calculations<Game>::p2l(const Array2D& AV, const Array2D& AU, Array2D& lAV, Array2D& lAU) {
  p2l_helper(AV, AU, lAV, lAU, [](const auto& x) { return eigen_util::logit(x); });
}

template <core::concepts::Game Game>
void Calculations<Game>::p2l_fast(const Array2D& AV, const Array2D& AU, Array2D& lAV,
                                  Array2D& lAU) {
  p2l_helper(AV, AU, lAV, lAU,
             [](const auto& mu_p) { return mu_p.unaryExpr(math::fast_coarse_logit); });
}

template <core::concepts::Game Game>
void Calculations<Game>::p2l(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW) {
  p2l_helper(Q, W, lQW, math::logit);
}

template <core::concepts::Game Game>
void Calculations<Game>::p2l_fast(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW) {
  p2l_helper(Q, W, lQW, math::fast_coarse_logit);
}

template <core::concepts::Game Game>
template <typename LogitFn>
void Calculations<Game>::p2l_helper(const Array2D& AV, const Array2D& AU, Array2D& lAV,
                                    Array2D& lAU, LogitFn&& logit_fn) {
  // operate only on column 0
  auto mu_p = AV.col(0);
  auto s_p = AU.col(0);
  auto mu_l = lAV.col(0);
  auto s_l = lAU.col(0);

  auto denom = (mu_p * (1 - mu_p)).eval();
  denom = denom * denom;
  auto inv_denom = eigen_util::invert(denom);

  auto logit_mu = logit_fn(mu_p);
  mu_l = logit_mu + s_p * (mu_p - 0.5f) * inv_denom;
  s_l = s_p * inv_denom;

  if (kNumPlayers == 2) {
    lAV.col(1) = -lAV.col(0);
    lAU.col(1) = lAU.col(0);
  }
}

template <core::concepts::Game Game>
template <typename LogitFn>
void Calculations<Game>::p2l_helper(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW,
                                    LogitFn&& logit_fn) {
  // operate only on column 0
  float mu_p = Q[0];
  float s_p = W[0];

  if (mu_p <= 0.f) {
    lQW[0] = util::Gaussian1D::neg_inf();
  } else if (mu_p >= 1.f) {
    lQW[0] = util::Gaussian1D::pos_inf();
  } else {
    float denom = mu_p * (1 - mu_p);
    denom = denom * denom;
    float inv_denom = 1.0f / denom;

    float mu_l = logit_fn(mu_p) + s_p * (mu_p - 0.5f) * inv_denom;
    float s_l = s_p * inv_denom;

    lQW[0] = util::Gaussian1D(mu_l, s_l);
  }

  if (kNumPlayers == 2) {
    lQW[1] = -lQW[0];
  }
}

template <core::concepts::Game Game>
void Calculations<Game>::l2p(const Array2D& lAV, const Array2D& lAU, Array2D& AV, Array2D& AU) {
  l2p_helper(lAV, lAU, AV, AU, [](const auto& x) { return eigen_util::sigmoid(x); });
}

template <core::concepts::Game Game>
void Calculations<Game>::l2p_fast(const Array2D& lAV, const Array2D& lAU, Array2D& AV,
                                  Array2D& AU) {
  l2p_helper(lAV, lAU, AV, AU,
             [](const auto& x) { return x.unaryExpr(math::fast_coarse_sigmoid); });
}

template <core::concepts::Game Game>
void Calculations<Game>::l2p(const Array2D& lAV, const Array2D& lAU, Array2D& AV) {
  l2p_helper(lAV, lAU, AV, [](const auto& x) { return eigen_util::sigmoid(x); });
}

template <core::concepts::Game Game>
void Calculations<Game>::l2p_fast(const Array2D& lAV, const Array2D& lAU, Array2D& AV) {
  l2p_helper(lAV, lAU, AV, [](const auto& x) { return x.unaryExpr(math::fast_coarse_sigmoid); });
}

template <core::concepts::Game Game>
template <typename Derived>
void Calculations<Game>::l2p(const Array1D& lAV, const Array1D& lAU,
                             Eigen::ArrayBase<Derived>& AV) {
  AV = eigen_util::sigmoid(lAV * (1.0f + kPiSquaredOver3 * lAU).rsqrt());
}

template <core::concepts::Game Game>
void Calculations<Game>::l2p_fast(const Array1D& lAV, const Array1D& lAU, Array1D& AV) {
  auto z = lAV * (1.0f + kPiSquaredOver3 * lAU).rsqrt();
  AV = z.unaryExpr(math::fast_coarse_sigmoid);
}

template <core::concepts::Game Game>
template <typename SigmoidFn>
void Calculations<Game>::l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV,
                                    Array2D& AU, SigmoidFn&& sigmoid_fn) {
  // operate only on column 0
  auto mu_l = lAV.col(0);
  auto s_l = lAU.col(0);
  auto mu_p = AV.col(0);
  auto s_p = AU.col(0);

  mu_p = sigmoid_fn(mu_l * (1.0f + kPiSquaredOver3 * s_l).rsqrt());

  auto x = sigmoid_fn(mu_l);
  auto y = x * (1.0f - x);
  s_p = s_l * y * y;

  if (kNumPlayers == 2) {
    AV.col(1) = 1.0f - AV.col(0);
    AU.col(1) = AU.col(0);
  }
}

template <core::concepts::Game Game>
template <typename SigmoidFn>
void Calculations<Game>::l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV,
                                    SigmoidFn&& sigmoid_fn) {
  // operate only on column 0
  auto mu_l = lAV.col(0);
  auto s_l = lAU.col(0);
  auto mu_p = AV.col(0);

  mu_p = sigmoid_fn(mu_l * (1.0f + kPiSquaredOver3 * s_l).rsqrt());

  if (kNumPlayers == 2) {
    AV.col(1) = 1.0f - AV.col(0);
  }
}

template <core::concepts::Game Game>
typename Calculations<Game>::ValueArray Calculations<Game>::scale_uncertainty(
  const ValueArray& V, const ValueArray& U01) {
  // operate only on column 0
  float v = V[0];
  float u01 = std::max(1.f, U01[0]);

  float u = u01 * v * (1.0f - v);
  ValueArray out;
  out[0] = u;

  if (kNumPlayers == 2) {
    out[1] = u;
  }
  return out;
}

template <core::concepts::Game Game>
typename Calculations<Game>::LocalActionValueArray Calculations<Game>::scale_uncertainty(
  const LocalActionValueArray& AV, const LocalActionValueArray& AU01) {
  // operate only on column 0
  auto v = AV.col(0);
  auto u01 = AU01.col(0).cwiseMax(1.f);

  auto u = u01 * v * (1.0f - v);
  LocalActionValueArray out(AV.rows(), AV.cols());
  out.col(0) = u;

  if (kNumPlayers == 2) {
    out.col(1) = u;
  }
  return out;
}

template <core::concepts::Game Game>
float Calculations<Game>::compute_beta(core::seat_index_t seat, const LocalPolicyArray& P,
                                       const ValueArray& V, const LocalActionValueArray& lAV,
                                       const LocalActionValueArray& lAU) {
  auto mu_l = lAV.col(seat);
  auto s_l = lAU.col(seat);
  const float v = V[seat];

  // Monotone function:
  //   F(beta) = P * l2p(mu_l + beta, lAU) - V
  // F is strictly increasing in beta, so we bracket and do safeguarded Newton.
  float minL = mu_l.minCoeff();
  float maxL = mu_l.maxCoeff();

  // Bracket: make all (mu_l + beta) very negative / very positive.
  constexpr float kSat = 12.0f;
  float lo = -maxL - kSat;
  float hi = -minL + kSat;

  float beta = 0.0f;
  if (beta < lo) beta = lo;
  if (beta > hi) beta = hi;

  constexpr int kIters = 16;      // conservative; usually converges faster
  constexpr float kTolF = 1e-7f;  // in value units (since sum(P)=1)

  LocalPolicyArray s = P;
  for (int it = 0; it < kIters; ++it) {
    float F = 0.0f;
    float dF = 0.0f;

    l2p(mu_l + beta, s_l, s);
    LocalPolicyArray Ps = P * s;
    F += Ps.sum();
    dF += (Ps * (1.0f - s)).sum();

    const float err = F - v;
    if (std::fabs(err) <= kTolF) break;

    // Maintain bracket (monotone increasing).
    if (err < 0.0f)
      lo = beta;
    else
      hi = beta;

    // Safeguarded Newton.
    const float c_newton = beta - err / (dF + 1e-12f);
    bool out_of_bounds = (c_newton < lo) || (c_newton > hi);
    beta = out_of_bounds ? (0.5f * (lo + hi)) : c_newton;
  }

  return beta;
}

template <core::concepts::Game Game>
float Calculations<Game>::compute_gamma(core::seat_index_t seat, const LocalPolicyArray& P,
                                        const LocalActionValueArray& AV, const ValueArray& U_beta) {
  auto av = AV.col(seat);
  float u_beta = U_beta[seat];

  if (u_beta == 0.f) {
    return 0.f;
  }

  RELEASE_ASSERT(u_beta > 0.f, "compute_gamma: U_beta must be nonnegative");
  float x = (P * av * (1.0f - av)).sum();
  RELEASE_ASSERT(x > 0.f, "compute_gamma: x must be positive when U_beta > 0");
  return u_beta / (x * x);
}

template <core::concepts::Game Game>
typename Calculations<Game>::ValueArray Calculations<Game>::compute_gamma_contribution(
  float gamma, float beta, const Array1D& pi, const Array1D& lAV, const Mask& not_E_mask) {
  ValueArray out = ValueArray::Zero();
  if (gamma == 0.f) {
    return out;
  }

  int n = not_E_mask.count();
  if (n == 0) {
    return out;
  }

  auto pi_m = eigen_util::mask_splice(pi, not_E_mask);
  auto lAV_m = eigen_util::mask_splice(lAV, not_E_mask);

  auto AV_m = eigen_util::sigmoid(kBeta * (lAV_m + beta));
  float x = (pi_m * AV_m * (1.0f - AV_m)).sum();
  float y = x * x * gamma;

  out.setConstant(y);
  return out;
}

template <core::concepts::Game Game>
void Calculations<Game>::Q_dot_product(core::seat_index_t seat, const Array1D& Q, const Array1D& pi,
                                       ValueArray& out) {
  out[seat] = dot_product_helper(Q, pi);
  if (kNumPlayers == 2) {
    out[1 - seat] = 1.0f - out[seat];
  }
}

template <core::concepts::Game Game>
void Calculations<Game>::W_dot_product(const Array1D& W, const Array1D& pi, ValueArray& out) {
  out[0] = dot_product_helper(W, pi);
  if (kNumPlayers == 2) {
    out[1] = out[0];
  }
}

template <core::concepts::Game Game>
float Calculations<Game>::dot_product_helper(const Array1D& X, const Array1D& pi) {
  auto support = (pi != 0.0f);
  bool full_support = support.all();

  float x_min;
  float x_max;
  if (full_support) {
    x_min = X.minCoeff();
    x_max = X.maxCoeff();
  } else {
    x_min = support.select(X, std::numeric_limits<float>::max()).minCoeff();
    x_max = support.select(X, std::numeric_limits<float>::lowest()).maxCoeff();
  }

  if (x_min == x_max) {
    return x_min;
  } else {
    return (X * pi).sum();
  }
}

}  // namespace beta0
