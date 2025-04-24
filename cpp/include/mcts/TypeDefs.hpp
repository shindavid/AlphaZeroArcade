#pragma once

#include <mcts/Constants.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace mcts {

using hash_shard_t = int8_t;

using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

struct MutexCv {
  std::mutex mutex;
  std::condition_variable cv;
};

using mutex_cv_vec_t = std::vector<MutexCv>;
using mutex_cv_vec_sptr_t = std::shared_ptr<mutex_cv_vec_t>;

}  // namespace mcts
