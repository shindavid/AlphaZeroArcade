#pragma once

#include <util/CppUtil.hpp>

namespace mcts {

enum Mode { kCompetitive, kTraining };

struct SearchThreadRegion {
  enum region_t {
    kAcquiringLazilyInitializedDataMutex,
    kLazyInit,
    kPureBackprop,
    kStandardBackprop,
    kShortCircuitBackprop,
    kMisc,
    kCheckingCache,
    kAcquiringBatchMutex,
    kWaitingUntilBatchReservable,
    kTensorizing,
    kIncrementingCommitCount,
    kWaitingForReservationProcessing,
    kVirtualBackprop,
    kUndoVirtualBackprop,
    kConstructingChildren,
    kPUCT,
    kAcquiringStatsMutex,
    kBackpropEvaluation,
    kMarkFullyAnalyzed,
    kExpand,
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

// The eval cache is split into 2^kCacheShardingFactor shards, each with its own mutex.
constexpr int8_t kCacheShardingFactor = 3;
static_assert(kCacheShardingFactor < 7);
constexpr int8_t kNumCacheShards = 1 << kCacheShardingFactor;

constexpr int kThreadWhitespaceLength = 50;  // for debug printing alignment
constexpr bool kEnableProfiling = IS_MACRO_ENABLED(PROFILE_MCTS);
constexpr bool kEnableVerboseProfiling = IS_MACRO_ENABLED(PROFILE_MCTS_VERBOSE);
constexpr bool kEnableSearchDebug = IS_MACRO_ENABLED(MCTS_DEBUG);
constexpr bool kEnableServiceDebug = IS_MACRO_ENABLED(MCTS_NN_SERVICE_DEBUG);

}  // namespace mcts
