#include "beta0/Calculations.hpp"

#include "util/Asserts.hpp"
#include "util/EigenUtil.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

namespace beta0 {

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

  auto denom_sqrt = mu_p * (1 - mu_p);
  RELEASE_ASSERT(!(denom_sqrt == 0.f).any(), "p2l_helper: denom has zero value(s)");
  auto inv_denom = 1.0f / (denom_sqrt * denom_sqrt);

  mu_l = logit_fn(mu_p) + s_p * (mu_p - 0.5f) * inv_denom;
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
  auto mu_p = Q[0];
  auto s_p = W[0];

  if (mu_p <= 0.f) {
    lQW[0] = util::Gaussian1D::neg_inf();
  } else if (mu_p >= 1.f) {
    lQW[0] = util::Gaussian1D::pos_inf();
  } else {
    auto denom = mu_p * (1 - mu_p);
    denom = denom * denom;
    auto inv_denom = 1.0f / denom;

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
void Calculations<Game>::l2p(const Array1D& lAV, const Array1D& lAU, Array1D& AV) {
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
float Calculations<Game>::compute_beta(const LocalPolicyArray& P, const ValueArray& V,
                                       const LocalActionValueArray& lAV,
                                       const LocalActionValueArray& lAU) {
  // operate only on column 0
  auto mu_l = lAV.col(0);
  auto s_l = lAU.col(0);
  const float v = V[0];

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
    }
  }
}

}  // namespace beta0
