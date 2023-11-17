#pragma once

#include <util/CppUtil.hpp>

#include <cstdint>

namespace mcts {

enum Mode { kCompetitive, kTraining };

enum TreeTraversalMode : int8_t { kPrefetchMode = 0, kSearchMode = 1, kNumTreeTraversalModes = 2 };

enum BackpropMode : int8_t { kTerminal, kNonterminal };

struct TreeTraversalThreadRegion {
  enum region_t {
    kBackprop,
    kBackpropWithVirtualUndo,
    kMisc,
    kCheckingCache,
    kWaitingUntilBatchReservable,
    kTensorizing,
    kIncrementingCommitCount,
    kWaitingForReservationProcessing,
    kVirtualBackprop,
    kPUCT,
    kEvaluate,
    kEvaluateUnset,
    kPrefetch,
    kSearch,
    kGetNextWorkItem,
    kWaitForSearchActivation,
    kWaitForEval,
    kWaitForEdge,
    kReset,
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
