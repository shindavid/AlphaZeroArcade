#include "util/Math.hpp"

#include <vector>

#include "util/Exceptions.hpp"
#include "util/StringUtil.hpp"

namespace math {

ExponentialDecay::ExponentialDecay(float value) : ExponentialDecay(value, value, 1.0) {}

ExponentialDecay::ExponentialDecay(float start_value, float end_value, float half_life)
    : start_value_(half_life ? start_value : end_value),
      end_value_(end_value),
      half_life_(half_life ? half_life : 1.0),
      decay_factor_(half_life_to_decay(half_life_)),
      cur_value_(start_value_) {}

}  // namespace math
