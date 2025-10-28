#include "util/Math.hpp"

namespace math {

namespace detail {

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
