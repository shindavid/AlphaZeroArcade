#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include <mcts/Constants.hpp>
#include <util/Profiler.hpp>

namespace mcts {

using search_thread_region_t = SearchThreadRegion::region_t;
#ifdef PROFILE_MCTS
using search_thread_profiler_t =
    util::Profiler<int(SearchThreadRegion::kNumRegions), kEnableVerboseProfiling>;
using nn_evaluation_service_profiler_t =
    util::Profiler<int(NNEvaluationServiceRegion::kNumRegions), kEnableVerboseProfiling>;
#else   // PROFILE_MCTS
using search_thread_profiler_t = util::DummyProfiler;
using nn_evaluation_service_profiler_t = util::DummyProfiler;
#endif  // PROFILE_MCTS

using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

struct MutexCv {
  std::mutex mutex;
  std::condition_variable cv;
};

using mutex_cv_vec_t = std::vector<MutexCv>;
using mutex_cv_vec_sptr_t = std::shared_ptr<mutex_cv_vec_t>;

}  // namespace mcts
