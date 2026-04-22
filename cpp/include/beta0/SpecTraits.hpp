#pragma once

#include "beta0/concepts/SpecConcept.hpp"

#include <Eigen/Core>

namespace beta0 {

/*
 * SpecTraits<Spec> centralizes the type aliases that depend on Spec's compile-time constants.
 *
 * Putting them here prevents repetition across Node data structures, BackupNNEvaluator, and the
 * Manager, and ensures consistent sizing everywhere.
 */
template <beta0::concepts::Spec Spec>
struct SpecTraits {
  using Game = Spec::Game;
  using ValueArray = Game::Types::ValueArray;

  static constexpr int kBackupHiddenDim = Spec::kBackupHiddenDim;
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  // Fixed-size Eigen column-vector for the backup-NN accumulator (hidden layer pre-activations).
  // Stored on each node; updated incrementally during search.
  using AccumulatorArray = Eigen::Array<float, kBackupHiddenDim, 1>;

  // Return type of BackupNNEvaluator::apply(): predicted (Q_parent, W_parent).
  struct QWPair {
    ValueArray Q;
    ValueArray W;
  };
};

}  // namespace beta0
