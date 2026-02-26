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
void Calculations<Game>::p2l(const Array1D& AV, const Array1D& AU, Array1D& lAV) {
  p2l_helper(AV, AU, lAV, [](const auto& x) { return eigen_util::logit(x); });
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

  RELEASE_ASSERT((s_p > 0.0f).all(), "AU must be non-negative (min: {})", s_p.minCoeff());

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
template <typename LogitFn>
void Calculations<Game>::p2l_helper(const Array1D& AV, const Array1D& AU, Array1D& lAV,
                                    LogitFn&& logit_fn) {
  const auto& mu_p = AV;;
  const auto& s_p = AU;;
  auto& mu_l = lAV;

  Mask zero_mask = s_p == 0.f;

  auto denom = (mu_p * (1 - mu_p)).eval();
  denom = denom * denom;
  denom = zero_mask.select(1.f, denom);
  auto inv_denom = eigen_util::invert(denom);

  auto logit_mu = logit_fn(mu_p);
  mu_l = logit_mu + s_p * (mu_p - 0.5f) * inv_denom;
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
  float min_lAU = lAU.minCoeff();
  if (min_lAU >= 0.f) {
    AV = eigen_util::sigmoid(lAV * (1.0f + kPiSquaredOver3 * lAU).rsqrt());
  } else {
    // we have +/- inf values in lAU; handle these correctly
    Mask mask = lAU >= 0.f;

    Array1D lAV_m = eigen_util::mask_splice(lAV, mask);
    Array1D lAU_m = eigen_util::mask_splice(lAU, mask);

    Array1D AV_m = eigen_util::sigmoid(lAV_m * (1.0f + kPiSquaredOver3 * lAU_m).rsqrt());

    eigen_util::mask_splice_assign(AV, mask, AV_m);

    int n = lAU.size();
    for (int i = 0; i < n; i++) {
      if (mask[i]) continue;

      float lAU_i = lAU[i];
      if (lAU_i == util::Gaussian1D::kVariancePosInf) {
        AV[i] = 1.f;
      } else {
        AV[i] = 0.f;
      }
    }
  }
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

  float min_s_l = s_l.minCoeff();
  if (min_s_l >= 0.f) {
    mu_p = sigmoid_fn(mu_l * (1.0f + kPiSquaredOver3 * s_l).rsqrt());

    auto x = mu_p * (1.0f - mu_p);
    s_p = s_l * x * x;
  } else {
    // we have +/- inf values in s_l; handle these correctly
    Mask mask = s_l >= 0.f;

    Array1D mu_l_m = eigen_util::mask_splice(mu_l, mask);
    Array1D s_l_m = eigen_util::mask_splice(s_l, mask);

    Array1D mu_p_m = sigmoid_fn(mu_l_m * (1.0f + kPiSquaredOver3 * s_l_m).rsqrt());

    auto x = mu_p_m * (1.0f - mu_p_m);
    Array1D s_p_m = s_l_m * x * x;

    eigen_util::mask_splice_assign(mu_p, mask, mu_p_m);
    eigen_util::mask_splice_assign(s_p, mask, s_p_m);

    int n = s_l.size();
    for (int i = 0; i < n; i++) {
      if (mask[i]) continue;

      float s_l_i = s_l[i];
      if (s_l_i == util::Gaussian1D::kVariancePosInf) {
        mu_p[i] = 1.f;
        s_p[i] = 0.f;
      } else {
        mu_p[i] = 0.f;
        s_p[i] = 0.0f;
      }
    }
  }

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

  float min_s_l = s_l.minCoeff();
  if (min_s_l >= 0.f) {
    mu_p = sigmoid_fn(mu_l * (1.0f + kPiSquaredOver3 * s_l).rsqrt());
  } else {
    // we have +/- inf values in s_l; handle these correctly
    Mask mask = s_l >= 0.f;

    Array1D mu_l_m = eigen_util::mask_splice(mu_l, mask);
    Array1D s_l_m = eigen_util::mask_splice(s_l, mask);

    Array1D mu_p_m = sigmoid_fn(mu_l_m * (1.0f + kPiSquaredOver3 * s_l_m).rsqrt());

    eigen_util::mask_splice_assign(mu_p, mask, mu_p_m);

    int n = s_l.size();
    for (int i = 0; i < n; i++) {
      if (mask[i]) continue;

      float s_l_i = s_l[i];
      if (s_l_i == util::Gaussian1D::kVariancePosInf) {
        mu_p[i] = 1.f;
      } else {
        mu_p[i] = 0.f;
      }
    }
  }

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
                                       const ValueArray& V, const LocalActionValueArray& AV,
                                       const LocalActionValueArray& AU) {
  constexpr float kPi2Over3 = math::kPi * math::kPi / 3.0f;

  auto av = AV.col(seat);
  auto au = AU.col(seat);
  const float v = V[seat];

  // p2l transform
  auto av_sq = av * av;
  auto onemav = 1.0f - av;
  auto onemav_sq = onemav * onemav;
  auto denom = av_sq * onemav_sq;  // (mu * (1-mu))^2

  auto lQ = eigen_util::logit(av) - (1.0f - 2.0f * av) * au / (2.0f * denom);
  auto lW = au / denom;

  // precompute loop-invariant values for l2p mean transform
  auto d = (1.0f + kPi2Over3 * lW).rsqrt();  // 1/sqrt(1 + pi^2 * lW / 3)
  auto s = lW.sqrt();                        // sqrt(lW)
  auto ds = d * s;
  auto dlQ = d * lQ;

  constexpr int kIters = 16;
  constexpr float kTolF = 1e-7f;

  float beta = 0.0f;

  for (int iter = 0; iter < kIters; ++iter) {
    auto z = dlQ + ds * beta;
    auto sig = eigen_util::sigmoid(z);
    auto Psig = P * sig;
    auto Psig_deriv = Psig * (1.0f - sig);

    float f = Psig.sum() - v;
    float f_prime = (Psig_deriv * ds).sum();

    if (f_prime < 1e-12f) break;

    float step = f / f_prime;
    beta -= step;

    if (std::abs(step) < kTolF) break;
  }

  return beta;
}

template <core::concepts::Game Game>
void Calculations<Game>::Q_dot_product(core::seat_index_t seat, const Array1D& Q, const Array1D& pi,
                                       ValueArray& out) {
  out[seat] = exact_dot_product(Q, pi);
  if (kNumPlayers == 2) {
    out[1 - seat] = 1.0f - out[seat];
  }
}

template <core::concepts::Game Game>
void Calculations<Game>::W_dot_product(const Array1D& W, const Array1D& pi, ValueArray& out) {
  out[0] = exact_dot_product(W, pi);
  if (kNumPlayers == 2) {
    out[1] = out[0];
  }
}

template <core::concepts::Game Game>
float Calculations<Game>::exact_dot_product(const Array1D& X, const Array1D& pi) {
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
