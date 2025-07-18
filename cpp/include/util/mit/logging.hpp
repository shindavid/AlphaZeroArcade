#pragma once

#include <util/CppUtil.hpp>

namespace mit {

constexpr bool kEnableDebugLogging = IS_MACRO_ENABLED(MIT_DEBUG_LOGGING);

}  // namespace mit

#define MIT_LOG(...) \
  if (mit::kEnableDebugLogging) { \
    LOG_INFO(__VA_ARGS__); \
  }
