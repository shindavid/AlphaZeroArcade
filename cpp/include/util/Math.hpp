#pragma once

#include <cmath>
#include <map>
#include <string>

namespace math {

using var_bindings_map_t = std::map<std::string, double>;

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
   */
  ExponentialDecay(float start_value, float end_value, float half_life);

  /*
   * "value" or "start->end:half_life"
   *
   * Each of "value", "start", "end", and "half_life" can be a general mathematical expression,
   * parsable via the tinyexpr library. Those expressions may include variables, in which case the
   * optional bindings argument must be provided.
   *
   * Examples:
   *
   * "4.32"             fixed 4.32, no decay
   * "10->2:0.4"        decay from 10 to 2, with a half-life of 0.4 moves
   * "5b->2b:sqrt(a)"   decay from 5b to 2b, with a half-life of sqrt(a) moves ("a" and "b" must be
   * bound in bindings)
   */
  static ExponentialDecay parse(const std::string& repr, const var_bindings_map_t& bindings);
  static ExponentialDecay parse(const std::string& repr) {
    return parse(repr, var_bindings_map_t{});
  }

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

/*
 * std::string expr("1 + sqrt(x+5)");
 * std::map<std::string, float> bindings;
 * bindings["x"] = 4;
 *
 * float y = parse_expression(expr, &bindings);  // y = 4
 */
double parse_expression(const char* expr, const var_bindings_map_t& bindings);

}  // namespace math

#include <inline/util/Math.inl>
