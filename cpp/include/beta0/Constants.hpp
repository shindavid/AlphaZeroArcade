#pragma once

#include <cstdint>

namespace beta0 {

enum ComputationCheckMethod : uint8_t { kAllowInf = 0, kAssertFinite = 1 };

}  // namespace beta0
