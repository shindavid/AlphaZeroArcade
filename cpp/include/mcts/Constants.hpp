#pragma once

#include <util/CppUtil.hpp>

namespace mcts {

enum Mode { kCompetitive, kTraining };

constexpr int kThreadWhitespaceLength = 50;  // for debug printing alignment
constexpr bool kEnableVerboseProfiling = IS_MACRO_ENABLED(PROFILE_MCTS_VERBOSE);
constexpr bool kEnableSearchDebug = IS_MACRO_ENABLED(MCTS_DEBUG);
constexpr bool kEnableServiceDebug = IS_MACRO_ENABLED(MCTS_NN_SERVICE_DEBUG);

constexpr int kNumHashShardsLog2 = 3;
constexpr int kNumHashShards = 1 << kNumHashShardsLog2;

}  // namespace mcts
