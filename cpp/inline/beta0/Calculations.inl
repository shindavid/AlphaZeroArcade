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

// template <core::concepts::Game Game>
// void Calculations<Game>::p2l(const Array1D& AV, const Array1D& AU, Array1D& lAV, Array1D& lAU) {
//   p2l_helper(AV, AU, lAV, lAU, [](const auto& x) { return eigen_util::logit(x); });
// }

template <core::concepts::Game Game>
template <typename LogitFn>
void Calculations<Game>::p2l_helper(const Array2D& AV, const Array2D& AU, Array2D& lAV,
                                    Array2D& lAU, LogitFn&& logit_fn) {
  // operate only on column 0
  auto mu_p = AV.col(0);
  auto s2_p = AU.col(0);
  auto mu_l = lAV.col(0);
  auto s2_l = lAU.col(0);

  RELEASE_ASSERT((s2_p > 0.0f).all(), "AU must be strictly positive (min: {})", s2_p.minCoeff());

  mu_l = logit_fn(mu_p);
  s2_l = p2l_var(mu_p, s2_p);

  if (kNumPlayers == 2) {
    lAV.col(1) = -lAV.col(0);
    lAU.col(1) = lAU.col(0);
  }
}

// template <core::concepts::Game Game>
// template <typename LogitFn>
// void Calculations<Game>::p2l_helper(const Array1D& AV, const Array1D& AU, Array1D& lAV,
//                                     Array1D& lAU, LogitFn&& logit_fn) {
//   lAV = logit_fn(AV);
//   lAU = p2l_var(AV, AU);
// }

template <core::concepts::Game Game>
template <typename LogitFn>
void Calculations<Game>::p2l_helper(const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW,
                                    LogitFn&& logit_fn) {
  // operate only on column 0
  float mu_p = Q[0];
  float s2_p = W[0];

  if (mu_p <= 0.f) {
    lQW[0] = util::Gaussian1D::neg_inf();
  } else if (mu_p >= 1.f) {
    lQW[0] = util::Gaussian1D::pos_inf();
  } else {
    float mu_l = logit_fn(mu_p);
    float s2_l = p2l_var(mu_p, s2_p);

    lQW[0] = util::Gaussian1D(mu_l, s2_l);
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
  float min_lAU = lAU.minCoeff();
  if (min_lAU >= 0.f) {
    AV = eigen_util::sigmoid(lAV);
  } else {
    // we have +/- inf values in lAU; handle these correctly
    Mask mask = lAU >= 0.f;

    Array1D lAV_m = eigen_util::mask_splice(lAV, mask);
    Array1D AV_m = eigen_util::sigmoid(lAV_m);
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
void Calculations<Game>::l2p_fast(const Array1D& lAV, Array1D& AV) {
  AV = lAV.unaryExpr(math::fast_coarse_sigmoid);
}

template <core::concepts::Game Game>
template <typename SigmoidFn>
void Calculations<Game>::l2p_helper(const Array2D& lAV, const Array2D& lAU, Array2D& AV,
                                    Array2D& AU, SigmoidFn&& sigmoid_fn) {
  // operate only on column 0
  auto mu_l = lAV.col(0);
  auto s2_l = lAU.col(0);
  auto mu_p = AV.col(0);
  auto s2_p = AU.col(0);

  float min_s2_l = s2_l.minCoeff();
  if (min_s2_l >= 0.f) {
    mu_p = sigmoid_fn(mu_l);
    s2_p = l2p_var(mu_l, s2_l);
  } else {
    // we have +/- inf values in s2_l; handle these correctly
    Mask mask = s2_l >= 0.f;

    Array1D mu_l_m = eigen_util::mask_splice(mu_l, mask);
    Array1D s2_l_m = eigen_util::mask_splice(s2_l, mask);

    Array1D mu_p_m = sigmoid_fn(mu_l_m);
    Array1D s2_p_m = l2p_var(mu_l_m, s2_l_m);

    eigen_util::mask_splice_assign(mu_p, mask, mu_p_m);
    eigen_util::mask_splice_assign(s2_p, mask, s2_p_m);

    int n = s2_l.size();
    for (int i = 0; i < n; i++) {
      if (mask[i]) continue;

      float s2_l_i = s2_l[i];
      if (s2_l_i == util::Gaussian1D::kVariancePosInf) {
        mu_p[i] = 1.f;
        s2_p[i] = 0.f;
      } else {
        mu_p[i] = 0.f;
        s2_p[i] = 0.0f;
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
  auto s2_l = lAU.col(0);
  auto mu_p = AV.col(0);

  float min_s2_l = s2_l.minCoeff();
  if (min_s2_l >= 0.f) {
    mu_p = sigmoid_fn(mu_l);
  } else {
    // we have +/- inf values in s2_l; handle these correctly
    Mask mask = s2_l >= 0.f;

    Array1D mu_l_m = eigen_util::mask_splice(mu_l, mask);
    Array1D mu_p_m = sigmoid_fn(mu_l_m);
    eigen_util::mask_splice_assign(mu_p, mask, mu_p_m);

    int n = s2_l.size();
    for (int i = 0; i < n; i++) {
      if (mask[i]) continue;

      float s2_l_i = s2_l[i];
      if (s2_l_i == util::Gaussian1D::kVariancePosInf) {
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
  float u01 = std::max(0.f, U01[0]);

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
  auto u01 = AU01.col(0).cwiseMax(0.f);

  auto u = u01 * v * (1.0f - v);
  LocalActionValueArray out(AV.rows(), AV.cols());
  out.col(0) = u;

  if (kNumPlayers == 2) {
    out.col(1) = u;
  }
  return out;
}

template <core::concepts::Game Game>
float Calculations<Game>::compute_beta(float Vs, const Array1D& pi, const Array1D& lAVs) {
  constexpr int kIters = 16;
  constexpr float kTolF = 1e-7f;

  float beta = 0.0f;

  for (int iter = 0; iter < kIters; ++iter) {
    auto z = lAVs + beta;
    auto sig = eigen_util::sigmoid(z);
    auto Psig = pi * sig;
    auto Psig_deriv = pi * sig * (1.0f - sig);

    float f = Psig.sum() - Vs;
    float f_prime = Psig_deriv.sum();

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

template <core::concepts::Game Game>
float Calculations<Game>::l2p_var(float lQ, float lW) {
  float sp = std::sqrt(lW);
  float std = (math::sigmoid(lQ + sp) - math::sigmoid(lQ - sp)) * 0.5f;
  return std * std;
}

template <core::concepts::Game Game>
auto Calculations<Game>::l2p_var(const Array1D& lQ, const Array1D& lW) {
  auto sp = lW.sqrt();
  auto std = (eigen_util::sigmoid(lQ + sp) - eigen_util::sigmoid(lQ - sp)) * 0.5f;
  Array1D out = std * std;
  return out;
}

template <core::concepts::Game Game>
float Calculations<Game>::p2l_var(float Q, float W) {
  float lQ = math::logit(Q);
  float target = 2.0f * std::sqrt(W);

  // Initial guess: linear approximation, clamped
  float QmQ = Q * (1.0f - Q);
  float sp = std::min(std::sqrt(W) / QmQ, 10.0f);

  for (int i = 0; i < 8; ++i) {
    float sig_plus = math::sigmoid(lQ + sp);
    float sig_minus = math::sigmoid(lQ - sp);
    float f = sig_plus - sig_minus - target;
    float f_prime = sig_plus * (1.0f - sig_plus) + sig_minus * (1.0f - sig_minus);

    if (f_prime < 1e-12f) break;

    float step = f / f_prime;
    sp -= step;
    sp = std::max(sp, 1e-8f);  // keep positive

    if (std::abs(step) < 1e-7f) break;
  }

  return std::min(sp * sp, 100.0f);
}

template <core::concepts::Game Game>
auto Calculations<Game>::p2l_var(const Array1D& Q, const Array1D& W) {
  Array1D out(Q.rows(), Q.cols());
  for (int i = 0; i < Q.size(); ++i) {
    out[i] = p2l_var(Q[i], W[i]);
  }
  return out;
}

}  // namespace beta0
