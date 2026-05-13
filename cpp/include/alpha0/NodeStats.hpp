#pragma once

#include "alpha0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
struct NodeStats {
  using Game = Spec::Game;
  using ValueArray = Game::Traits::ValueArray;
  using player_bitset_t = Game::Traits::player_bitset_t;

  int total_count() const { return RN + VN; }

  // Mirrors beta0::NodeStats::Q() — a method-style accessor (rather than direct field access)
  // so generic search-side code (e.g. SearchLog) can call `.Q()` uniformly across paradigms.
  ValueArray Q() const { return Q_; }
  float Q(core::seat_index_t seat) const { return Q_(seat); }
  void setQ(const ValueArray& q) { Q_ = q; }

  ValueArray Q_;     // excludes virtual loss
  ValueArray Q_sq;   // excludes virtual loss
  int RN = 0;        // real count
  int VN = 0;        // virtual count

  // TODO: generalize these fields to utility lower/upper bounds
  player_bitset_t provably_winning;
  player_bitset_t provably_losing;
};

}  // namespace alpha0
