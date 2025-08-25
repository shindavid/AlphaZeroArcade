#pragma once

#include "util/CppUtil.hpp"

#include <cstdint>

namespace nnet {

using hash_shard_t = int8_t;

constexpr int kNumHashShardsLog2 = 3;
constexpr int kNumHashShards = 1 << kNumHashShardsLog2;

constexpr bool kEnableServiceDebug = IS_DEFINED(MCTS_NN_SERVICE_DEBUG);

}  // namespace nnet
