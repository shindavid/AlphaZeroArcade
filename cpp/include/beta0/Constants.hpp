#pragma once

#include <cstdint>

namespace beta0 {

static constexpr float kBeta = 1.702f;  // logistic approximation constant
static constexpr float kInvBeta = 1.0f / kBeta;

static constexpr float kLambda = 0.f;  // pi contribution to w_{ij}
static constexpr bool kDisableZMargining = false;

enum ComputationCheckMethod : uint8_t { kAllowInf = 0, kAssertFinite = 1 };

}  // namespace beta0
