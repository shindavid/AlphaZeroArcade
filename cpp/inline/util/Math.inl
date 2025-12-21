#include "util/Math.hpp"

#include <algorithm>

namespace math {

namespace detail {

// ----------------------------
// Logit LUT
// ----------------------------
struct LogitLUT {
  static constexpr int kSize = 256;
  static constexpr float kMuMin = 1e-4f;
  static constexpr float kMuMax = 1.0f - 1e-4f;
  static constexpr float kRange = kMuMax - kMuMin;
  static constexpr float kScale = (kSize - 1) / kRange;

  alignas(64) float table[kSize];

  LogitLUT() {
    for (int i = 0; i < kSize; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(kSize - 1);  // 0..1
      float mu = kMuMin + t * kRange;
      // exact logit for the LUT construction
      table[i] = std::log(mu / (1.0f - mu));
    }
  }
};

inline const LogitLUT& get_logit_lut() {
  static const LogitLUT lut;
  return lut;
}

// ----------------------------
// Sigmoid LUT
// ----------------------------
struct SigmoidLUT {
  static constexpr int kSize = 256;
  static constexpr float kXMin = -8.0f;
  static constexpr float kXMax = 8.0f;
  static constexpr float kRange = kXMax - kXMin;
  static constexpr float kScale = (kSize - 4) / kRange;

  alignas(64) float table[kSize];

  SigmoidLUT() {
    // Exact left tail
    table[0] = 0.0f;

    for (int i = 1; i < kSize - 2; ++i) {
      float t = float(i - 1) / float(kSize - 4);
      float x = kXMin + t * kRange;
      float s = 1.0f / (1.0f + std::exp(-x));
      table[i] = s;
    }

    // Exact right tail (two slots)
    table[kSize - 2] = 1.0f;
    table[kSize - 1] = 1.0f;
  }
};

inline const SigmoidLUT& get_sigmoid_lut() {
  static const SigmoidLUT lut;
  return lut;
}

// ----------------------------
// Phi LUT
// ----------------------------
struct PhiLUT {
  static constexpr int kSize = 256;
  static constexpr float kXMin = -4.0f;
  static constexpr float kXMax = 4.0f;
  static constexpr float kRange = kXMax - kXMin;
  static constexpr float kScale = (kSize - 4) / kRange;

  // We set table[0], table[kSize-1], and table[kSize-2] to exact values to allow
  // fast_coarse_batch_normal_cdf() to handle out-of-bounds inputs without branching.
  alignas(64) float table[kSize];

  PhiLUT() {
    // Exact left tail
    table[0] = 0.0f;

    for (int i = 1; i < kSize - 2; ++i) {
      float t = float(i - 1) / (kSize - 4);
      float x = kXMin + t * kRange;
      table[i] = normal_cdf(x);
    }

    // Exact right tail (two slots)
    table[kSize - 2] = 1.0f;
    table[kSize - 1] = 1.0f;
  }
};

inline const PhiLUT& get_phi_lut() {
  static const PhiLUT lut;
  return lut;
}

inline float pdf_exact(float x) {
  // standard normal pdf
  constexpr float inv_sqrt_2pi = 0.39894228040143267794f;
  return inv_sqrt_2pi * std::exp(-0.5f * x * x);
}

// Very fast inverse-CDF guess (logistic approximation).
inline float invphi_logistic_guess(float p) {
  // Phi^{-1}(p) ~ log(p/(1-p)) / 1.702
  constexpr float a = 1.702f;
  // Clamp away from 0/1
  p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
  return std::log(p / (1.0f - p)) / a;
}

// ----------------------------
// Inverse Phi LUT
// ----------------------------
struct InvPhiLUT {
  static constexpr int kSize = 256;
  static constexpr float kPMin = 1e-6f;
  static constexpr float kPMax = 1.0f - 1e-6f;
  static constexpr float kRange = kPMax - kPMin;
  static constexpr float kScale = (kSize - 1) / kRange;

  alignas(64) float table[kSize];

  InvPhiLUT() {
    for (int i = 0; i < kSize; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(kSize - 1);  // [0,1]
      float p = kPMin + t * kRange;

      // Initial guess
      float x = invphi_logistic_guess(p);

      // Two Newton refinements using exact CDF/PDF.
      // x_{new} = x - (Phi(x) - p) / phi(x)
      for (int it = 0; it < 2; ++it) {
        float fx = normal_cdf(x) - p;
        float d = pdf_exact(x);
        // Guard against tiny derivative
        x -= fx / std::max(d, 1e-12f);
      }

      table[i] = x;
    }
  }
};

inline const InvPhiLUT& get_invphi_lut() {
  static const InvPhiLUT lut;
  return lut;
}

// Fast inverse Phi using LUT + linear interpolation
inline float fast_invphi_lut(float p) {
  const auto& lut = get_invphi_lut();

  // Clamp to LUT domain
  p = std::clamp(p, InvPhiLUT::kPMin, InvPhiLUT::kPMax);

  float t = (p - InvPhiLUT::kPMin) * InvPhiLUT::kScale;  // in [0, kSize-1)
  int idx = static_cast<int>(t);

  if (idx >= InvPhiLUT::kSize - 1) return lut.table[InvPhiLUT::kSize - 1];
  if (idx < 0) return lut.table[0];

  float frac = t - static_cast<float>(idx);

  float a = lut.table[idx];
  float b = lut.table[idx + 1];
  return a + (b - a) * frac;
}

// log Phi(-z) for z >= 0  (upper-tail log-prob)
inline double logPhiNeg_moderate(double z_nonneg) {
  const double t = z_nonneg * M_SQRT1_2;
  return std::log(0.5) + std::log(std::erfc(t));
}

// log Phi(z), stable across ranges when |z| <= THRESH
inline double logPhi_moderate(double z) {
  if (z < 0) {
    const double t = -z * M_SQRT1_2;
    return std::log(0.5) + std::log(std::erfc(t));
  } else {
    const double s = logPhiNeg_moderate(z);
    return std::log1p(-std::exp(s));
  }
}

// log odds = log(Phi(z) / Phi(-z))
inline double log_odds_normal(double z) {
  static constexpr double THRESH = 8.0;
  if (z < -THRESH) {
    const double t = -z;
    const double inv = 1.0 / t;
    const double R = inv - inv * inv * inv + 3 * std::pow(inv, 5) - 15 * std::pow(inv, 7);
    const double logphi = -0.5 * t * t - 0.5 * std::log(2.0 * M_PI);
    return logphi + std::log(R);
  }
  if (z > THRESH) {
    return -log_odds_normal(-z);  // symmetry
  }
  return logPhi_moderate(z) - logPhi_moderate(-z);
}

}  // namespace detail

inline float fast_coarse_logit(float mu) {
  const auto& lut = detail::get_logit_lut();
  using LUT = detail::LogitLUT;

  float x = std::clamp(mu, LUT::kMuMin, LUT::kMuMax);

  float t = (x - LUT::kMuMin) * LUT::kScale;
  int idx = static_cast<int>(t);
  if (idx >= LUT::kSize - 1) idx = LUT::kSize - 2;

  float frac = t - static_cast<float>(idx);

  float a = lut.table[idx];
  float b = lut.table[idx + 1];

  return a + (b - a) * frac;
}

inline float fast_coarse_sigmoid(float x) {
  const auto& lut = detail::get_sigmoid_lut();
  using SigmoidLUT = detail::SigmoidLUT;

  constexpr float xmin = SigmoidLUT::kXMin;
  constexpr float scale = SigmoidLUT::kScale;
  constexpr float tmax = float(SigmoidLUT::kSize - 2);

  // Map x to LUT coordinate, with a 1.0 offset because table[0] is the left tail.
  float t = 1.0f + (x - xmin) * scale;
  t = std::clamp(t, 0.0f, tmax);

  int idx = static_cast<int>(t);
  float frac = t - float(idx);

  float a = lut.table[idx];
  float b = lut.table[idx + 1];
  return a + (b - a) * frac;
}

inline void fast_coarse_batch_normal_cdf(const float* __restrict x, int n, float* __restrict y) {
  const auto& lut = detail::get_phi_lut();
  using PhiLUT = detail::PhiLUT;

  constexpr float xmin = PhiLUT::kXMin;
  constexpr float scale = PhiLUT::kScale;
  constexpr float tmax = float(PhiLUT::kSize - 2);

  for (int i = 0; i < n; ++i) {
    float t = 1.0f + (x[i] - xmin) * scale;
    t = std::clamp(t, 0.0f, tmax);

    int idx = t;
    float frac = t - idx;

    float a = lut.table[idx];
    float b = lut.table[idx + 1];

    y[i] = a + (b - a) * frac;
  }
}

inline void fast_coarse_batch_inverse_normal_cdf_clamped_range(float p0, const float* __restrict p,
                                                               const float* __restrict c, int n,
                                                               float* __restrict y, float eps) {
  float rmin[n];
  float rmax[n];

  for (int i = 0; i < n; ++i) {
    float inv_denom = 1.0f / (p0 + p[i]);
    rmin[i] = (p0 - eps) * inv_denom;
    rmax[i] = (p0 + eps) * inv_denom;
  }

  for (int i = 0; i < n; ++i) {
    float y_min = detail::fast_invphi_lut(rmin[i]);
    float y_max = detail::fast_invphi_lut(rmax[i]);
    y[i] = std::clamp(c[i], y_min, y_max);
  }
}

inline ExponentialDecay::ExponentialDecay(float value) : ExponentialDecay(value, value, 1.0) {}

inline ExponentialDecay::ExponentialDecay(float start_value, float end_value, float half_life)
    : start_value_(half_life ? start_value : end_value),
      end_value_(end_value),
      half_life_(half_life ? half_life : 1.0),
      decay_factor_(half_life_to_decay(half_life_)),
      cur_value_(start_value_) {}

inline double normal_cdf_logit_diff(double z_new, double z_old) {
  return detail::log_odds_normal(z_new) - detail::log_odds_normal(z_old);
}

}  // namespace math
