#pragma once

#include "beta0/concepts/SpecConcept.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct NodeStats {
  using Game = Spec::Game;
  using ValueArray = Game::Types::ValueArray;
  using player_bitset_t = Game::Types::player_bitset_t;

  int total_count() const { return RN + VN; }

  ValueArray Q;     // excludes virtual loss
  ValueArray Q_sq;  // excludes virtual loss
  ValueArray W;     // uncertainty estimate, per-player
  int RN = 0;       // real count
  int VN = 0;       // virtual count

  // TODO: generalize these fields to utility lower/upper bounds
  player_bitset_t provably_winning;
  player_bitset_t provably_losing;
};

}  // namespace beta0
