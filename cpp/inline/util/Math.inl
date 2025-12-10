#include "util/Math.hpp"

namespace math {

namespace detail {

inline float phi_exact(float x) {
  // Used only for LUT initialization.
  // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
  return 0.5f * (1.0f + std::erff(x * 0.7071067811865475244f));
}

struct PhiLUT {
  static constexpr int kSize = 257;
  static constexpr float kXMin = -4.0f;
  static constexpr float kXMax = 4.0f;
  static constexpr float kRange = kXMax - kXMin;
  static constexpr float kScale = (kSize - 1) / kRange;
  static constexpr float kInvScale = 1.0f / kScale;

  alignas(64) float table[kSize];

  PhiLUT() {
    for (int i = 0; i < kSize; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(kSize - 1);  // [0,1]
      float x = kXMin + t * kRange;
      table[i] = phi_exact(x);
    }
  }
};

inline const PhiLUT& get_phi_lut() {
  // Thread-safe static init since C++11.
  static const PhiLUT lut;
  return lut;
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

inline void fast_coarse_batch_normal_cdf(const float* __restrict x, int n, float* __restrict y) {
  const auto& lut = detail::get_phi_lut();
  using PhiLUT = detail::PhiLUT;

  constexpr float xmin = PhiLUT::kXMin;
  constexpr float xmax = PhiLUT::kXMax;
  constexpr float scale = PhiLUT::kScale;

  // Encourage vectorization where possible.
  // Note: LUT access is effectively a gather; full SIMD speedups may be modest,
  // but this loop is still very cheap.
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
  for (int i = 0; i < n; ++i) {
    float xi = x[i];

    if (xi <= xmin) {
      y[i] = 0.0f;
      continue;
    }
    if (xi >= xmax) {
      y[i] = 1.0f;  // If you *really* want 0 outside range, change this to 0.0f.
      continue;
    }

    float t = (xi - xmin) * scale;  // [0, 255)
    int idx = static_cast<int>(t);

    // Defensive clamp
    if (idx < 0) idx = 0;
    if (idx > PhiLUT::kSize - 2) idx = PhiLUT::kSize - 2;

    float frac = t - static_cast<float>(idx);

    float a = lut.table[idx];
    float b = lut.table[idx + 1];
    y[i] = a + (b - a) * frac;
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
