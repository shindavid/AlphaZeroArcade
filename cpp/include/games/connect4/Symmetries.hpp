#pragma once

#include "core/BasicTypes.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputFrame.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace c4 {

struct Symmetries {
  static c4::Game::Types::SymmetryMask get_mask(const InputFrame& frame);

  static void apply(InputFrame& frame, group::element_t sym);

  template <eigen_util::concepts::FTensor Tensor>
  static void apply(Tensor& tensor, group::element_t sym, core::action_mode_t);

  static group::element_t get_canonical_symmetry(const InputFrame& frame);
};

}  // namespace c4

#include "inline/games/connect4/Symmetries.inl"
