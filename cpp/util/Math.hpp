#pragma once

#include <cmath>

namespace math {

inline float half_life_to_decay(float half_life) {
  return half_life > 0 ? exp(log(0.5) / half_life) : 1.0;
}

}  // namespace math
