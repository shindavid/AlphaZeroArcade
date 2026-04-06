#pragma once

#include "games/hex/Game.hpp"
#include "games/hex/InputFrame.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace hex {

struct Symmetries {
  static Game::Types::SymmetryMask get_mask(const InputFrame& frame);

  static void apply(InputFrame& frame, group::element_t sym);

  template <eigen_util::concepts::FTensor Tensor>
  static void apply(Tensor& tensor, group::element_t sym, const InputFrame& frame);

  template <eigen_util::concepts::FTensor Tensor>
  static void apply(Tensor& tensor, group::element_t sym);

  static group::element_t get_canonical_symmetry(const InputFrame& frame);
};

}  // namespace hex

#include "inline/games/hex/Symmetries.inl"
