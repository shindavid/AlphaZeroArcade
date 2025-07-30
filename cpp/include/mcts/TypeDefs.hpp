#pragma once

#include "util/mit/mit.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace mcts {

using hash_shard_t = int8_t;

using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

using mutex_vec_t = std::vector<mit::mutex>;
using mutex_vec_sptr_t = std::shared_ptr<mutex_vec_t>;

}  // namespace mcts
