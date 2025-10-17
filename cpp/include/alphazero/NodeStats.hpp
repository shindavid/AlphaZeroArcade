#pragma once

#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats {
  using Game = EvalSpec::Game;
  using ValueArray = Game::Types::ValueArray;
  using player_bitset_t = Game::Types::player_bitset_t;

  int total_count() const { return RN + VN; }

  ValueArray Q;     // excludes virtual loss
  ValueArray Q_sq;  // excludes virtual loss
  int RN = 0;       // real count
  int VN = 0;       // virtual count

  // TODO: generalize these fields to utility lower/upper bounds
  player_bitset_t provably_winning;
  player_bitset_t provably_losing;
};

}  // namespace alpha0
