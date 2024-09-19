#pragma once

#include <cmath>
#include <map>
#include <string>

namespace math {

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
  void step() { cur_value_ += (end_value_ - cur_value_) * decay_factor_; }

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

template<typename T>
constexpr T round_up_to_nearest_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace math
