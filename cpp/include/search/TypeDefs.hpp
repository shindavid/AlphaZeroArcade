#pragma once

#include "util/AllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace search {

using hash_shard_t = int8_t;

using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

using mutex_vec_t = std::vector<mit::mutex>;
using mutex_vec_sptr_t = std::shared_ptr<mutex_vec_t>;

using node_pool_index_t = util::pool_index_t;
using edge_pool_index_t = util::pool_index_t;

enum expansion_state_t : int8_t {
  kNotExpanded,
  kMidExpansion,
  kPreExpanded,  // used when evaluating all children when computing AV-targets
  kExpanded
};

}  // namespace search
