#pragma once

#include <chrono>

#include <util/CppUtil.hpp>
#include <util/Profiler.hpp>

namespace mcts {

enum Mode {
  kCompetitive,
  kTraining
};

struct SearchThreadRegion {
  enum region_t {
    kCheckVisitReady = 0,
    kAcquiringLazilyInitializedDataMutex = 1,
    kLazyInit = 2,
    kBackpropOutcome = 3,
    kPerformEliminations = 4,
    kMisc = 5,
    kCheckingCache = 6,
    kAcquiringBatchMutex = 7,
    kWaitingUntilBatchReservable = 8,
    kTensorizing = 9,
    kIncrementingCommitCount = 10,
    kWaitingForReservationProcessing = 11,
    kVirtualBackprop = 12,
    kConstructingChildren = 13,
    kPUCT = 14,
    kAcquiringStatsMutex = 15,
    kBackpropEvaluation = 16,
    kMarkFullyAnalyzed = 17,
    kEvaluateAndExpand = 18,
    kEvaluateAndExpandUnset = 19,
    kEvaluateAndExpandPending = 20,
    kNumRegions = 21
  };
};

struct NNEvaluationServiceRegion {
  enum region_t {
    kWaitingUntilBatchReady = 0,
    kWaitingForFirstReservation = 1,
    kWaitingForLastReservation = 2,
    kWaitingForCommits = 3,
    kCopyingCpuToGpu = 4,
    kEvaluatingNeuralNet = 5,
    kCopyingToPool = 6,
    kAcquiringCacheMutex = 7,
    kFinishingUp = 8,
    kNumRegions = 9
  };
};

constexpr bool kEnableProfiling = IS_MACRO_ENABLED(PROFILE_MCTS);
constexpr bool kEnableVerboseProfiling = IS_MACRO_ENABLED(PROFILE_MCTS_VERBOSE);
constexpr bool kEnableThreadingDebug = IS_MACRO_ENABLED(MCTS_THREADING_DEBUG);

using search_thread_region_t = SearchThreadRegion::region_t;
#ifdef PROFILE_MCTS
using search_thread_profiler_t = util::Profiler<int(SearchThreadRegion::kNumRegions), kEnableVerboseProfiling>;
using nn_evaluation_service_profiler_t = util::Profiler<int(NNEvaluationServiceRegion::kNumRegions), kEnableVerboseProfiling>;
#else  // PROFILE_MCTS
using search_thread_profiler_t = util::DummyProfiler;
using nn_evaluation_service_profiler_t = util::DummyProfiler;
#endif  // PROFILE_MCTS

using child_index_t = int;
using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

}  // namespace mcts
