#pragma once

#include <chrono>

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

using move_number_t = int;
using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

}  // namespace mcts
