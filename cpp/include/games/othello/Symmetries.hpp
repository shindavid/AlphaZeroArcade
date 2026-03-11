#pragma once

#include "games/othello/Game.hpp"

#include "core/BasicTypes.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace othello {

struct Symmetries {
  static Game::Types::SymmetryMask get_mask(const Game::State& state);

  static void apply(Game::State& state, group::element_t sym);

  template <eigen_util::concepts::FTensor Tensor>
  static void apply(Tensor& tensor, group::element_t sym, core::action_mode_t = 0);

  static group::element_t get_canonical_symmetry(const Game::State& state);
};

}  // namespace othello

#include "inline/games/othello/Symmetries.inl"
