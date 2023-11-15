#pragma once

#include <util/CppUtil.hpp>

#include <cstdint>

namespace mcts {

enum Mode { kCompetitive, kTraining };

enum TreeTraversalMode : int8_t { kPrefetchMode = 0, kSearchMode = 1, kNumTreeTraversalModes = 2 };

struct TreeTraversalThreadRegion {
  enum region_t {
    kCheckVisitReady,
    kAcquiringLazilyInitializedDataMutex,
    kLazyInit,
    kPureBackprop,
    kBackpropWithVirtualUndo,
    kMisc,
    kCheckingCache,
    kAcquiringBatchMutex,
    kWaitingUntilBatchReservable,
    kTensorizing,
    kIncrementingCommitCount,
    kWaitingForReservationProcessing,
    kVirtualBackprop,
    kConstructingChildren,
    kPUCT,
    kAcquiringStatsMutex,
    kBackpropEvaluation,
    kMarkFullyAnalyzed,
    kEvaluate,
    kEvaluateUnset,
    kEvaluatePending,
    kNumRegions
  };
};

struct NNEvaluationServiceRegion {
  enum region_t {
    kWaitingUntilBatchReady,
    kWaitingForFirstReservation,
    kWaitingForLastReservation,
    kWaitingForCommits,
    kCopyingCpuToGpu,
    kEvaluatingNeuralNet,
    kCopyingToPool,
    kAcquiringCacheMutex,
    kFinishingUp,
    kNumRegions
  };
};

constexpr bool kEnableProfiling = IS_MACRO_ENABLED(PROFILE_MCTS);
constexpr bool kEnableVerboseProfiling = IS_MACRO_ENABLED(PROFILE_MCTS_VERBOSE);
constexpr bool kEnableDebug = IS_MACRO_ENABLED(MCTS_DEBUG);

}  // namespace mcts
