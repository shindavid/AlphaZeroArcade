#pragma once

#include <cstdint>

namespace beta0 {

enum OutcomeCertainty : uint8_t {
  kUncertain = 0,
  kCertainDraw = 1,
  kCertainWin = 2,
  kCertainLoss = 3
};

}  // namespace beta0
