#pragma once

#include "mcts/SearchResults.hpp"
#include "nnet/NNEvaluation.hpp"
#include "search/ManagerParams.hpp"
#include "search/NodeBase.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

namespace mcts {

/*
 * A Node consists of n=3 main data members:
 *
 * StableData: write-once data that is fixed for the lifetime of the node
 * Stats: values that get updated throughout MCTS via backpropagation
 * Edge[]: edges to children nodes, needed for tree traversal
 *
 * The last one, Edge[], is represented by a single index into a pool of edges.
 *
 * During MCTS, multiple search threads will try to read and write these values. Thread-safety is
 * achieved in a high-performance manner through mutexes and condition variables.
 */
template <typename Traits>
class Node : public search::NodeBase<Traits> {
 public:
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  using ManagerParams = search::ManagerParams<Traits>;
  using NNEvaluation = nnet::NNEvaluation<Traits>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using ValueArray = Game::Types::ValueArray;
  using ValueTensor = Game::Types::ValueTensor;
  using SearchResults = mcts::SearchResults<Traits>;
  using player_bitset_t = Game::Types::player_bitset_t;

  using NodeBase = search::NodeBase<Traits>;
  using StableData = NodeBase::StableData;

  // Generally, we acquire this->mutex() when reading or writing to this->stats_. There are some
  // exceptions on reads, when we read a single atomically-writable member of stats_.
  struct Stats {
    int total_count() const { return RN + VN; }
    void init_q(const ValueArray&, bool pure);
    void update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                              const player_bitset_t& all_actions_provably_losing,
                              int num_expanded_children, bool cp_has_winning_move,
                              const StableData&);

    ValueArray Q;     // excludes virtual loss
    ValueArray Q_sq;  // excludes virtual loss
    int RN = 0;       // real count
    int VN = 0;       // virtual count

    // TODO: generalize these fields to utility lower/upper bounds
    player_bitset_t provably_winning;
    player_bitset_t provably_losing;
  };

  using NodeBase::NodeBase;

  void write_results(const ManagerParams& params, group::element_t inv_sym,
                     SearchResults& results) const;

  template <typename MutexProtectedFunc>
  void update_stats(MutexProtectedFunc);

  // stats() returns a reference to the stats object, WITHOUT acquiring the mutex. In order to use
  // this function properly, the caller must ensure that one of the following is true:
  //
  // 1. The context is single-threaded,
  //
  // or,
  //
  // 2. The usage of the stats reference falls within the scope of the node's mutex,
  //
  // or,
  //
  // 3. The caller is ok with the possibility of a race-condition with a writer.
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }

  // Acquires the mutex and returns a copy of the stats object.
  Stats stats_safe() const;

  template <typename PolicyTransformFunc>
  void load_eval(NNEvaluation* eval, PolicyTransformFunc);

  // NO-OP in release builds, checks various invariants in debug builds
  void validate_state() const;

 private:
  Stats stats_;
};

}  // namespace mcts

#include "inline/mcts/Node.inl"
