#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace math {

enum finiteness_t : int8_t { kFinite, kPosInf, kNegInf };

template <typename T>
finiteness_t get_finiteness(T x) {
  throw std::runtime_error("get_finiteness() no longer valid after we enabled -ffast-math");
}

inline float normal_cdf(float x) { return 0.5f * (1.0f + std::erff(x * 0.7071067811865475244f)); }

// Approximates log(x) for x in the range (0, 1)
float fast_coarse_log_less_than_1(float x);

// Approximates logit(x).
//
// Returns a value close to log(x / (1 - x)), but uses a fast approximation that is less accurate
// when x is very close to 0 or 1.
float fast_coarse_logit(float x);

// Approximates sigmoid(x). Returns value in the range (0, 1).
//
// Returns a value close to 1 / (1 + exp(-x)), but uses a fast approximation that is less accurate
// when x is very large positive or negative.
float fast_coarse_sigmoid(float x);

// Very-fast coarse approximation of normal CDF for batch processing.
//
// Sets y[i] ~= Phi(x[i]) for i in [0, n).
//
// Under the hood, use a piecewise linear approximation with 256 segments over the range
// [-4, +4], and clamp to 0 or 1 outside that range.
void fast_coarse_batch_normal_cdf(const float* __restrict x, int n, float* __restrict y);

inline float sigmoid(float x) { return 0.5f * (std::tanh(0.5f * x) + 1.0f); }  // avoids overflow

// Very-fast coarse approximation of a specialized clamped-range inverse normal CDF calculation for
// batch processing.
//
// Define
//
// f(x) = inverse_normal_cdf(x)
//
// with f(x) taking the values -inf at x<=0 and +inf at x>=1.
//
// For each i in [0, n), define R[i] to be the set of ratios q0 / (q0 + q[i]), where q0 is a float
// in the range [p0 - eps, p0 + eps], and q[i] is a float in the range [p[i] - eps, p[i] + eps].
//
// Sets y[i] ~= clamp(c[i], m[i], M[i]), where
//
// m[i] = min_{r in R[i]} f(r)
// M[i] = max_{r in R[i]} f(r)
void fast_coarse_batch_inverse_normal_cdf_clamped_range(float p0, const float* __restrict p,
                                                        const float* __restrict c, int n,
                                                        float* __restrict y, float eps = 0.01f);

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
  void step(float k) {
    cur_value_ = end_value_ + (cur_value_ - end_value_) * std::pow(decay_factor_, k);
  }

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
