#pragma once

#include <util/CppUtil.hpp>

namespace mcts {

enum Mode { kCompetitive, kTraining };

struct SearchThreadRegion {
  enum region_t {
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
