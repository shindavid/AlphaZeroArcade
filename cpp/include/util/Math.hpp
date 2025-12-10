#pragma once

#include <cmath>
#include <cstdint>

namespace math {

enum finiteness_t : int8_t { kFinite, kPosInf, kNegInf };

template <typename T>
finiteness_t get_finiteness(T x) {
  return std::isfinite(x) ? kFinite : (x > 0 ? kPosInf : kNegInf);
}

template <typename T>
inline auto normal_cdf(T x) {
  return float(0.5) * erfc(-x * M_SQRT1_2);
}

// Very-fast coarse approximation of normal CDF for batch processing.
//
// Under the hood, use a piecewise linear approximation with 256 segments over the range
// [-4, +4], and clamp to 0 or 1 outside that range.
inline void fast_coarse_batch_normal_cdf(const float* __restrict x, int n, float* __restrict y);

// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

constexpr inline uint64_t constexpr_pow(uint64_t base, uint32_t exp) {
  return (exp == 0) ? 1 : base * constexpr_pow(base, exp - 1);
}

/*
 * We frequently want to decay some parameter from A to B, using a half-life specified in moves.
 * This class is used to facilitate such calculations in an efficient manner using a clear
 * interface.
 */
class ExponentialDecay {
 public:
  ExponentialDecay(float value = 0);  // use this for a non-decaying static value

  /*
   * Starts at start_value. Every half_life moves, gets 50% of the way closer to end_value.
   *
   * If half_life is zero, then this acts just like a fixed value of end_value.
   */
  ExponentialDecay(float start_value, float end_value, float half_life);

  void reset() { cur_value_ = start_value_; }
  float value() const { return cur_value_; }
  void step() { cur_value_ = end_value_ + (cur_value_ - end_value_) * decay_factor_; }

 private:
  const float start_value_;
  const float end_value_;
  const float half_life_;
  const float decay_factor_;

  float cur_value_;
};

inline float half_life_to_decay(float half_life) {
  return half_life > 0 ? exp(log(0.5) / half_life) : 1.0;
}

template <typename T>
constexpr T round_up_to_nearest_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// Returns the logit difference of the normal CDF for two z-scores. In other words, returns
// log(p / (1 - p)) - log(q / (1 - q)), where p = Phi(z_new) and q = Phi(z_old).
//
// Performs this calculation in a numerically stable manner. Even if z_new and z_old are large
// positive or negative numbers, the result should be accurate as long as the logit difference
// itself is not extremely large or small.
double normal_cdf_logit_diff(double z_new, double z_old);

}  // namespace math

#include "inline/util/Math.inl"
