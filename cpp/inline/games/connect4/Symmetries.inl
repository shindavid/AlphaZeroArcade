#include "games/connect4/Symmetries.hpp"

#include "core/BasicTypes.hpp"
#include "core/DefaultCanonicalizer.hpp"
#include "games/connect4/InputFrame.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"

#include <bit>

namespace c4 {

inline Game::Types::SymmetryMask Symmetries::get_mask(const InputFrame& frame) {
  Game::Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Symmetries::apply(InputFrame& frame, group::element_t sym) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      frame.full_mask = std::byteswap(frame.full_mask << 8);
      frame.cur_player_mask = std::byteswap(frame.cur_player_mask << 8);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

template <eigen_util::concepts::FTensor Tensor>
inline void Symmetries::apply(Tensor& t, group::element_t sym, core::game_phase_t) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      Tensor u = eigen_util::reverse(t, 0);
      t = u;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline group::element_t Symmetries::get_canonical_symmetry(const InputFrame& frame) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<InputFrame, Symmetries>;
  return DefaultCanonicalizer::get(frame);
}

}  // namespace c4
