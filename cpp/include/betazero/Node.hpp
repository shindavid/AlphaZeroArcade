#pragma once

#include "core/NodeBase.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

namespace beta0 {

// beta0::Node has everything that alpha0::Node has, plus uncertainty tracking.
template <core::concepts::EvalSpec EvalSpec>
class Node : public core::NodeBase<EvalSpec> {
 public:
  using Game = EvalSpec::Game;
  using ValueArray = Game::Types::ValueArray;
  using player_bitset_t = Game::Types::player_bitset_t;

  using NodeBase = core::NodeBase<EvalSpec>;
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
    float W = 0;      // dynamic uncertainty

    // TODO: generalize these fields to utility lower/upper bounds
    player_bitset_t provably_winning;
    player_bitset_t provably_losing;
  };

  using NodeBase::NodeBase;

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

 private:
  Stats stats_;
};

}  // namespace beta0

#include "inline/betazero/Node.inl"
