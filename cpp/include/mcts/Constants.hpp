#pragma once

#include <util/CppUtil.hpp>

namespace mcts {

enum Mode { kCompetitive, kTraining };

constexpr int kThreadWhitespaceLength = 50;  // for debug printing alignment
constexpr bool kEnableSearchDebug = IS_DEFINED(MCTS_DEBUG);
constexpr bool kEnableServiceDebug = IS_DEFINED(MCTS_NN_SERVICE_DEBUG);

constexpr int kNumHashShardsLog2 = 3;
constexpr int kNumHashShards = 1 << kNumHashShardsLog2;

}  // namespace mcts
