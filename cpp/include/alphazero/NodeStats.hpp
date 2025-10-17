#pragma once

#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats {
  using Game = EvalSpec::Game;
  using ValueArray = Game::Types::ValueArray;
  using player_bitset_t = Game::Types::player_bitset_t;

  int total_count() const { return RN + VN; }
  void update_q(const ValueArray& q, const ValueArray& q_sq, bool pure);
  void update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                            const player_bitset_t& all_actions_provably_losing,
                            bool cp_has_winning_move, bool all_edges_expanded,
                            core::seat_index_t seat);

  ValueArray Q;     // excludes virtual loss
  ValueArray Q_sq;  // excludes virtual loss
  int RN = 0;       // real count
  int VN = 0;       // virtual count

  // TODO: generalize these fields to utility lower/upper bounds
  player_bitset_t provably_winning;
  player_bitset_t provably_losing;
};

}  // namespace alpha0

#include "inline/alphazero/NodeStats.inl"
