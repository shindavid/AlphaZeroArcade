#pragma once

#include <util/CppUtil.hpp>

namespace mcts {

constexpr bool kEnableProfiling = IS_MACRO_ENABLED(PROFILE_MCTS);
constexpr bool kEnableVerboseProfiling = IS_MACRO_ENABLED(PROFILE_MCTS_VERBOSE);
constexpr bool kEnableThreadingDebug = IS_MACRO_ENABLED(MCTS_THREADING_DEBUG);

}  // namespace mcts
